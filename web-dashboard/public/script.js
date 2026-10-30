const socket = io();

const ctx = document
  .getElementById("waveChart")
  .getContext("2d");

const chart = new Chart(ctx, {
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label: "Heart Signal",
      data: [],
      borderColor: "#dc2626",
      backgroundColor: "rgba(220,38,38,0.15)",
fill: true,
      borderWidth: 2,
      tension: 0.2
    }]
  },
  options: {
    animation: false,
    responsive: true,
    scales: {
      y: {
        beginAtZero: false  // Auto-scale mengikuti data aktual
      }
    }
  }
});

socket.on("sensorData", (data) => {

  document.getElementById("current").innerText =
    data.value;

  document.getElementById("bpm").innerText =
    data.bpm;

  document.getElementById("prediction").innerText =
    data.status;

  chart.data.labels.push("");

  chart.data.datasets[0].data.push(data.value);

  if(chart.data.labels.length > 100){
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }

  chart.update();
});