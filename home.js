document.getElementById("detect_btn").onclick = function() {
    var text_content = document.getElementById("symptoms").value;
    if (text_content === "") {
        document.getElementById("prediction_output").innerHTML = "Invalid command, please enter valid symptoms.";
    } else {
        document.getElementById("prediction_output").innerHTML = "Model working..."; 
        setTimeout(function() {
            document.getElementById("prediction_output").innerHTML = "Prediction: Placeholder for disease prediction.";
        }, 1000); 
    }
};