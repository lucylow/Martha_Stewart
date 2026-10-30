(function () {
  "use strict";

  var canvas = document.getElementById("portfolio-chart");
  var rangeButtons = Array.prototype.slice.call(document.querySelectorAll("[data-range]"));
  var fallback = document.getElementById("chart-fallback");
  var chart = null;
  var series = {
    "7D": { labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], values: [100, 100.8, 100.4, 101.5, 101.2, 102.1, 102.6] },
    "30D": { labels: ["Oct 1", "Oct 6", "Oct 11", "Oct 16", "Oct 21", "Oct 26", "Oct 30"], values: [100, 101.2, 100.7, 102.4, 103.1, 102.8, 104.2] },
    "90D": { labels: ["Aug 1", "Aug 15", "Sep 1", "Sep 15", "Oct 1", "Oct 15", "Oct 30"], values: [100, 98.9, 101.4, 100.8, 102.2, 103.1, 104.2] }
  };

  if (canvas && typeof Chart !== "undefined") {
    chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: series["30D"].labels,
        datasets: [{
          data: series["30D"].values,
          borderColor: "#176a4a",
          backgroundColor: "rgba(23, 106, 74, 0.09)",
          borderWidth: 2,
          pointRadius: 2.5,
          pointHoverRadius: 4,
          pointBackgroundColor: "#ffffff",
          pointBorderColor: "#176a4a",
          pointBorderWidth: 1.5,
          lineTension: 0.32,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        legend: { display: false },
        tooltips: {
          displayColors: false,
          callbacks: { label: function (item) { return "Sample index: " + item.yLabel; } }
        },
        scales: {
          xAxes: [{ gridLines: { display: false, drawBorder: false }, ticks: { fontColor: "#68776e", maxTicksLimit: 7, maxRotation: 0 } }],
          yAxes: [{ gridLines: { color: "#edf1ed", drawBorder: false, zeroLineColor: "#edf1ed" }, ticks: { fontColor: "#68776e", maxTicksLimit: 5, padding: 8, callback: function (value) { return value.toFixed ? value.toFixed(0) : value; } } }]
        },
        layout: { padding: { top: 6, right: 8, bottom: 0, left: 0 } }
      }
    });
    if (fallback) fallback.hidden = true;
  }

  rangeButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      var range = button.getAttribute("data-range");
      if (!series[range]) return;
      if (chart) {
        chart.data.labels = series[range].labels;
        chart.data.datasets[0].data = series[range].values;
        chart.update();
      } else if (fallback) {
        fallback.textContent = "Illustrative " + range + " portfolio performance. Sample index values: " + series[range].values.join(", ") + ".";
      }
      rangeButtons.forEach(function (item) {
        var active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-pressed", active ? "true" : "false");
      });
    });
  });

  var menuButton = document.querySelector(".nav-toggle");
  var navigation = document.getElementById("primary-nav");
  if (menuButton && navigation) {
    function setMenuOpen(open) {
      navigation.classList.toggle("is-open", open);
      menuButton.setAttribute("aria-expanded", open ? "true" : "false");
      menuButton.setAttribute("aria-label", open ? "Close navigation menu" : "Open navigation menu");
    }
    menuButton.addEventListener("click", function () {
      setMenuOpen(menuButton.getAttribute("aria-expanded") !== "true");
    });
    navigation.addEventListener("click", function (event) {
      if (event.target.closest("a")) setMenuOpen(false);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") {
        setMenuOpen(false);
        menuButton.focus();
      }
    });
  }

  var search = document.getElementById("research-search");
  var areaFilter = document.getElementById("research-area");
  var table = document.querySelector(".research-table");
  var count = document.getElementById("research-count");
  var emptyRow = document.getElementById("research-empty");
  var exportButton = document.getElementById("export-research");
  var exportStatus = document.getElementById("export-status");
  var studyRows = table ? Array.prototype.slice.call(table.querySelectorAll("tbody tr[data-area]")) : [];

  function visibleStudies() {
    return studyRows.filter(function (row) { return !row.hidden; });
  }

  function updateStudies() {
    var query = search ? search.value.trim().toLowerCase() : "";
    var area = areaFilter ? areaFilter.value : "all";
    var visibleCount = 0;
    studyRows.forEach(function (row) {
      var matchesArea = area === "all" || row.getAttribute("data-area") === area;
      var matchesQuery = !query || row.textContent.toLowerCase().indexOf(query) !== -1;
      row.hidden = !(matchesArea && matchesQuery);
      if (!row.hidden) visibleCount += 1;
    });
    if (emptyRow) emptyRow.hidden = visibleCount !== 0;
    if (count) count.textContent = "Showing " + visibleCount + " sample " + (visibleCount === 1 ? "study" : "studies");
  }

  if (search) search.addEventListener("input", updateStudies);
  if (areaFilter) areaFilter.addEventListener("change", updateStudies);

  function csvCell(value) {
    var text = String(value || "").replace(/\r?\n/g, " ");
    if (/^[=+\-@\t\r]/.test(text)) text = "'" + text;
    return '"' + text.replace(/"/g, '""') + '"';
  }

  if (exportButton) {
    exportButton.addEventListener("click", function () {
      var rows = [["Sample data only — synthetic research workspace"], ["Study", "Detail", "Area", "Updated", "Status"]];
      visibleStudies().forEach(function (row) {
        rows.push([
          row.querySelector(".study-name").textContent.trim(),
          row.querySelector(".study-detail").textContent.trim(),
          row.cells[1].textContent.trim(),
          row.cells[2].textContent.trim(),
          row.querySelector(".status-tag").textContent.trim()
        ]);
      });
      var csv = "\ufeff" + rows.map(function (row) { return row.map(csvCell).join(","); }).join("\r\n");
      var url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
      var link = document.createElement("a");
      link.href = url;
      link.download = "martha-stewart-sample-research.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      if (exportStatus) exportStatus.textContent = "Exported " + visibleStudies().length + " sample studies to CSV.";
    });
  }
}());
