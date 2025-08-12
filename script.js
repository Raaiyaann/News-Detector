const API_ENDPOINT = "https://your-backend-api.com/predict"; // Replace with your backend

document.getElementById("predictForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const newsUrl = document.getElementById("newsUrl").value;
    const resultDiv = document.getElementById("result");

    resultDiv.className = "hidden";
    resultDiv.textContent = "Checking...";

    try {
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: newsUrl })
        });

        const data = await response.json();

        if (data.prediction === "hoax") {
            resultDiv.className = "result-true";
            resultDiv.textContent = "⚠ This news might be a hoax!";
        } else if (data.prediction === "real") {
            resultDiv.className = "result-false";
            resultDiv.textContent = "✅ This news seems real.";
        } else {
            resultDiv.className = "";
            resultDiv.textContent = "❓ Unable to determine. Please try again.";
        }
    } catch (error) {
        resultDiv.className = "";
        resultDiv.textContent = "Error connecting to server.";
    }
});
