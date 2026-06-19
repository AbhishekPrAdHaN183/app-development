// Global instances for charts
let forecastChartInstance = null;
let categoryChartInstance = null;
let validationChartInstance = null;
let transactionsList = [];

// Initialize Dashboard when DOM loads
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initDragAndDrop();
    
    // Initial fetch of all dashboard telemetry
    refreshDashboardTelemetry();
});

// --- Tab Navigation Setup ---
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const tabPanes = document.querySelectorAll(".tab-pane");

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const targetTab = item.getAttribute("data-tab");
            
            navItems.forEach(nav => nav.classList.remove("active"));
            tabPanes.forEach(pane => pane.classList.remove("active"));
            
            item.classList.add("active");
            document.getElementById(targetTab).classList.add("active");
            
            // Re-render charts or run simulator logic when switching tabs
            if (targetTab === "simulator-tab") {
                updateElasticity();
            } else if (targetTab === "dashboard-tab") {
                refreshDashboardTelemetry();
            }
        });
    });
}

// --- Fetch & Refresh Analytics Telemetry ---
async function refreshDashboardTelemetry() {
    try {
        const statsRes = await fetch("/api/stats");
        if (!statsRes.ok) throw new Error("Failed to load dashboard metrics");
        
        const statsData = await statsRes.json();
        updateKPICards(statsData.summary, statsData.model_status);
        updateModelBadge(statsData.model_status);
        
        if (statsData.model_status.trained) {
            renderFeatureImportances(statsData.model_status.feature_importance);
        }
        
        // Load transaction table and charts
        await fetchHistoricalAndForecastData();
        renderCategoryDistributionChart(statsData.categories);
        
    } catch (err) {
        console.error("Dashboard Loading Error:", err);
    }
}

// Update high-level KPI visual statistics
function updateKPICards(summary, modelStatus) {
    document.getElementById("statRevenue").innerText = `$${summary.total_revenue.toLocaleString()}`;
    document.getElementById("statDateRange").innerHTML = `<i class="fa-solid fa-calendar-days"></i> ${summary.date_range}`;
    document.getElementById("statUnits").innerText = summary.total_units.toLocaleString();
    document.getElementById("statTxCount").innerText = `${summary.transaction_count.toLocaleString()} transaction logs`;
    document.getElementById("statPromoCount").innerText = `${summary.promo_transactions.toLocaleString()} promo sales`;

    if (modelStatus.trained) {
        document.getElementById("statR2").innerText = modelStatus.metrics.r2.toFixed(4);
        document.getElementById("statAlgoName").innerText = modelStatus.algorithm.replace("_", " ").toUpperCase();
        document.getElementById("statRMSE").innerText = `${modelStatus.metrics.rmse.toFixed(1)} units`;
    } else {
        document.getElementById("statR2").innerText = "N/A";
        document.getElementById("statAlgoName").innerText = "No model trained";
        document.getElementById("statRMSE").innerText = "N/A";
    }
}

// Update model status header badge
function updateModelBadge(modelStatus) {
    const badge = document.getElementById("modelBadge");
    const badgeText = document.getElementById("modelBadgeText");
    
    if (modelStatus.trained) {
        badge.className = "model-badge";
        badgeText.innerText = `${modelStatus.algorithm.replace("_", " ").toUpperCase()} MODEL ACTIVE`;
        badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${badgeText.innerText}`;
    } else {
        badge.className = "model-badge untrained";
        badgeText.innerText = "MODEL UNTRAINED (CLICK MODEL CONSOLE)";
        badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${badgeText.innerText}`;
    }
}

// Render feature importance bar indicators
function renderFeatureImportances(importances) {
    const listContainer = document.getElementById("featureImportanceList");
    if (!importances || importances.length === 0) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-brain"></i>
                <p>No feature importances found.</p>
            </div>`;
        return;
    }

    let html = "";
    importances.forEach(item => {
        // Format feature names nicely
        const cleanName = item.feature.replace(/_/g, " ").replace("sales", "Sales").replace("lag", "Lag");
        const percentage = Math.round(item.importance * 100);
        
        html += `
            <div class="feature-bar-container">
                <div class="feature-info">
                    <span class="feature-name">${cleanName}</span>
                    <span class="feature-value">${percentage}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-bar" style="width: ${percentage}%"></div>
                </div>
            </div>`;
    });
    listContainer.innerHTML = html;
}

// --- Fetch & Render main forecasting graph ---
async function fetchHistoricalAndForecastData() {
    try {
        const histRes = await fetch("/api/historical-data");
        const forecastRes = await fetch("/api/forecast");
        
        const historical = histRes.ok ? await histRes.json() : [];
        const forecast = forecastRes.ok ? await forecastRes.json() : [];
        
        renderForecastChart(historical, forecast);
        populateTransactionsTable(historical);
        
    } catch (err) {
        console.error("Forecasting plot retrieval failed:", err);
    }
}

// Render forecast line graph combining historical sales and prediction outputs
function renderForecastChart(historical, forecast) {
    const ctx = document.getElementById("forecastChart").getContext("2d");
    
    // Clear previous chart instance
    if (forecastChartInstance) forecastChartInstance.destroy();
    
    // Group historical transactions by date to show daily total quantities
    const dates = historical.map(item => item.date);
    const actualSales = historical.map(item => item.units_sold);
    
    // Create prediction dataset offsetting it chronologically
    const forecastDates = forecast.map(item => item.date);
    const forecastSales = forecast.map(item => item.predicted_demand);
    
    // Align datasets
    const combinedLabels = [...dates, ...forecastDates];
    
    // Pad historical actuals for forecast days with null, and forecast predictions for historical days with null
    const actualData = [...actualSales, ...Array(forecastSales.length).fill(null)];
    
    // Connecting actuals and forecasts smoothly
    const predData = [...Array(actualSales.length - 1).fill(null), actualSales[actualSales.length - 1], ...forecastSales];
    
    // Limit displaying to the last 90 historical days + forecast to avoid clutter
    const maxVisiblePoints = 90 + forecast.length;
    const startIdx = Math.max(0, combinedLabels.length - maxVisiblePoints);
    
    const visibleLabels = combinedLabels.slice(startIdx);
    const visibleActual = actualData.slice(startIdx);
    const visiblePred = predData.slice(startIdx);
    
    // Setup gradients
    const actualGrd = ctx.createLinearGradient(0, 0, 0, 300);
    actualGrd.addColorStop(0, "rgba(79, 172, 254, 0.4)");
    actualGrd.addColorStop(1, "rgba(79, 172, 254, 0.0)");
    
    const forecastGrd = ctx.createLinearGradient(0, 0, 0, 300);
    forecastGrd.addColorStop(0, "rgba(0, 242, 254, 0.4)");
    forecastGrd.addColorStop(1, "rgba(0, 242, 254, 0.0)");

    forecastChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: visibleLabels,
            datasets: [
                {
                    label: "Historical Actuals",
                    data: visibleActual,
                    borderColor: "#4facfe",
                    borderWidth: 2,
                    backgroundColor: actualGrd,
                    fill: true,
                    tension: 0.2,
                    pointRadius: visibleActual.length > 50 ? 0 : 3
                },
                {
                    label: "Predicted Future Demand",
                    data: visiblePred,
                    borderColor: "#00f2fe",
                    borderWidth: 3,
                    borderDash: [5, 5],
                    backgroundColor: forecastGrd,
                    fill: true,
                    tension: 0.2,
                    pointRadius: visiblePred.length > 50 ? 0 : 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "#e8eaed", font: { family: "Plus Jakarta Sans" } }
                }
            },
            scales: {
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#909bb0", font: { family: "Plus Jakarta Sans" } },
                    title: { display: true, text: "Daily Demand (Units Sold)", color: "#e8eaed" }
                },
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.02)" },
                    ticks: { color: "#909bb0", font: { family: "Plus Jakarta Sans" }, maxRotation: 45 }
                }
            }
        }
    });
}

// Render category distribution pie chart
function renderCategoryDistributionChart(categories) {
    const ctx = document.getElementById("categoryChart").getContext("2d");
    if (categoryChartInstance) categoryChartInstance.destroy();
    
    if (!categories || categories.length === 0) return;
    
    const labels = categories.map(c => c.category);
    const revenues = categories.map(c => c.revenue);
    
    categoryChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: revenues,
                backgroundColor: [
                    "rgba(0, 242, 254, 0.7)",
                    "rgba(79, 172, 254, 0.7)",
                    "rgba(138, 43, 226, 0.7)",
                    "rgba(16, 185, 129, 0.7)"
                ],
                borderColor: "#0f111a",
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: "#e8eaed", font: { family: "Plus Jakarta Sans", size: 11 } }
                }
            }
        }
    });
}

// --- Elasticity Simulator ---
async function updateElasticity() {
    const price = parseFloat(document.getElementById("simPrice").value);
    const promo = document.getElementById("simPromo").checked;
    
    document.getElementById("simPriceVal").innerText = `$${price.toFixed(2)}`;
    
    try {
        const res = await fetch("/api/predict-elasticity", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ unit_price: price, is_promo: promo })
        });
        
        if (!res.ok) throw new Error("Simulation prediction failed");
        
        const data = await res.json();
        
        document.getElementById("simResultUnits").innerText = data.predicted_units_sold.toFixed(1);
        document.getElementById("simResultRevenue").innerText = `$${data.estimated_revenue.toLocaleString()}`;
        
        // Dynamically compute pricing impact details
        computeElasticityAnalysis(price, promo, data.predicted_units_sold);
        
    } catch (err) {
        console.error("Simulation Error:", err);
    }
}

// Output elasticity narrative explanations
function computeElasticityAnalysis(price, promo, predictedUnits) {
    const container = document.getElementById("simInsightText");
    
    // Baseline checks: simple rules based on pricing triggers
    let priceText = "";
    if (price > 120) {
        priceText = "This high price point severely restricts demand volume. Sales are constrained to premium/inelastic buyers only.";
    } else if (price < 30) {
        priceText = "At this low price point, demand is highly active. Be mindful of potential stockout risks if supply is constrained.";
    } else {
        priceText = "Pricing is balanced within a moderate demand zone, optimizing margins vs volume.";
    }
    
    let promoText = "";
    if (promo) {
        promoText = " The active promotional campaign creates a strong demand uplift, shifting the elasticity curve upward and driving high volume.";
    } else {
        promoText = " Operating under normal pricing structures without promo boosters means volume relies strictly on organic brand demand.";
    }
    
    container.innerHTML = `<i class="fa-solid fa-circle-nodes" style="color: var(--accent-cyan)"></i> ${priceText} ${promoText}`;
}

// --- Model Console Retraining Handler ---
async function triggerRetraining(e) {
    e.preventDefault();
    
    const btnText = document.getElementById("btnTrainText");
    const btnSpinner = document.getElementById("btnTrainSpinner");
    const btn = document.getElementById("btnTrain");
    
    btnText.classList.add("hidden");
    btnSpinner.classList.remove("hidden");
    btn.disabled = true;
    
    const config = {
        algorithm: document.getElementById("trainAlgo").value,
        lag_days: parseInt(document.getElementById("trainLags").value),
        rolling_window: parseInt(document.getElementById("trainWindow").value),
        test_size: parseFloat(document.getElementById("trainTestSize").value) / 100
    };
    
    try {
        const res = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config)
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Model training failed");
        }
        
        const data = await res.json();
        
        // Show validation chart, remove overlay
        document.getElementById("validationChartLoading").classList.add("hidden");
        document.getElementById("validationChart").classList.remove("hidden");
        
        renderValidationChart(data.test_eval);
        
        // Alert user on success
        alert(`Model retrained successfully using ${data.algorithm.toUpperCase()}! R²: ${data.metrics.r2.toFixed(4)}, RMSE: ${data.metrics.rmse.toFixed(2)}`);
        
        // Refresh general dashboard telemetry
        refreshDashboardTelemetry();
        
    } catch (err) {
        alert("Retraining Error: " + err.message);
    } finally {
        btnText.classList.remove("hidden");
        btnSpinner.classList.add("hidden");
        btn.disabled = false;
    }
}

// Render validation evaluation comparison actuals vs predicted chart
function renderValidationChart(testEval) {
    const ctx = document.getElementById("validationChart").getContext("2d");
    if (validationChartInstance) validationChartInstance.destroy();
    
    const labels = testEval.map(item => item.date);
    const actuals = testEval.map(item => item.actual);
    const predictions = testEval.map(item => item.predicted);
    
    validationChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Validation Actuals",
                    data: actuals,
                    borderColor: "rgba(255,255,255,0.4)",
                    borderWidth: 2,
                    pointRadius: 2,
                    fill: false
                },
                {
                    label: "Validation Predictions",
                    data: predictions,
                    borderColor: "#a252ff",
                    borderWidth: 2.5,
                    pointRadius: 2,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "#e8eaed", font: { family: "Plus Jakarta Sans" } }
                }
            },
            scales: {
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#909bb0", font: { family: "Plus Jakarta Sans" } }
                },
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.02)" },
                    ticks: { color: "#909bb0", font: { family: "Plus Jakarta Sans" } }
                }
            }
        }
    });
}

// --- Data Manager Operations ---
function populateTransactionsTable(data) {
    transactionsList = data;
    filterTransactions(); // Load filtered data (loads all initially since input is empty)
}

function filterTransactions() {
    const searchVal = document.getElementById("txSearch").value.toLowerCase().trim();
    const tbody = document.getElementById("txTableBody");
    
    // Group transactions by category/product for granular list or show raw
    let filtered = transactionsList;
    if (searchVal !== "") {
        filtered = transactionsList.filter(tx => 
            tx.product_name.toLowerCase().includes(searchVal) || 
            tx.product_id.toLowerCase().includes(searchVal) ||
            tx.category.toLowerCase().includes(searchVal)
        );
    }
    
    // Display latest first
    const visibleTx = [...filtered].reverse().slice(0, 50); // Cap table at 50 elements for speed
    
    if (visibleTx.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="loading-td">No transactions match search criteria</td></tr>`;
        return;
    }
    
    let html = "";
    visibleTx.forEach(tx => {
        const promoBadge = tx.is_promo 
            ? '<span class="badge badge-promo">Promo</span>' 
            : '<span class="badge badge-normal">Standard</span>';
        
        html += `
            <tr>
                <td>${tx.date}</td>
                <td><strong>${tx.product_id}</strong></td>
                <td>${tx.product_name}</td>
                <td>${tx.category}</td>
                <td>${tx.units_sold}</td>
                <td>$${tx.unit_price.toFixed(2)}</td>
                <td>${promoBadge}</td>
            </tr>`;
    });
    tbody.innerHTML = html;
}

// Handle manual logging form submission
async function logManualTransaction(e) {
    e.preventDefault();
    
    const payload = {
        date: document.getElementById("manualDate").value,
        product_id: document.getElementById("manualId").value.trim(),
        product_name: document.getElementById("manualName").value.trim(),
        category: document.getElementById("manualCategory").value,
        units_sold: parseInt(document.getElementById("manualUnits").value),
        unit_price: parseFloat(document.getElementById("manualPrice").value),
        is_promo: document.getElementById("manualPromo").checked
    };
    
    try {
        const res = await fetch("/api/add-transaction", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Failed to log transaction");
        }
        
        alert("Sale logged successfully!");
        document.getElementById("manualTxForm").reset();
        refreshDashboardTelemetry();
        
    } catch (err) {
        alert("Logging Error: " + err.message);
    }
}

// --- Drag and Drop File Ingestion ---
function initDragAndDrop() {
    const dropArea = document.getElementById("fileDropArea");
    const fileInput = document.getElementById("csvFile");
    
    if (!dropArea) return;
    
    ["dragenter", "dragover"].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.classList.add("dragover");
        }, false);
    });
    
    ["dragleave", "drop"].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.classList.remove("dragover");
        }, false);
    });
    
    dropArea.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            triggerFileSelect(files[0]);
        }
    }, false);
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length) {
        triggerFileSelect(files[0]);
    }
}

function triggerFileSelect(file) {
    if (!file.name.endsWith(".csv")) {
        alert("Only CSV files are supported.");
        clearFileSelection();
        return;
    }
    
    document.getElementById("fileDropArea").classList.add("hidden");
    const infoPanel = document.getElementById("selectedFileInfo");
    infoPanel.classList.remove("hidden");
    document.getElementById("selectedFileName").innerText = file.name;
    document.getElementById("btnUpload").disabled = false;
}

function clearFileSelection() {
    document.getElementById("csvFile").value = "";
    document.getElementById("fileDropArea").classList.remove("hidden");
    document.getElementById("selectedFileInfo").classList.add("hidden");
    document.getElementById("btnUpload").disabled = true;
    
    const status = document.getElementById("uploadStatusPanel");
    status.className = "upload-alert hidden";
}

// Handle bulk CSV uploading
async function uploadCSV(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById("csvFile");
    if (!fileInput.files.length) return;
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    
    const statusPanel = document.getElementById("uploadStatusPanel");
    const statusText = document.getElementById("uploadStatusText");
    const btn = document.getElementById("btnUpload");
    
    btn.disabled = true;
    statusPanel.className = "upload-alert info";
    statusText.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Parsing report...`;
    statusPanel.classList.remove("hidden");
    
    try {
        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });
        
        const data = await res.json();
        
        if (!res.ok) {
            throw new Error(data.error || "Failed to process upload");
        }
        
        statusPanel.className = "upload-alert success";
        statusText.innerText = data.message;
        
        // Success alert, reload dashboard
        setTimeout(() => {
            clearFileSelection();
            refreshDashboardTelemetry();
        }, 1500);
        
    } catch (err) {
        statusPanel.className = "upload-alert error";
        statusText.innerText = "Upload Error: " + err.message;
        btn.disabled = false;
    }
}
