const canvas = document.getElementById("lossCanvas");
const ctx = canvas.getContext("2d");
const win0_height = canvas.height * 0.8
const win1_height = canvas.height * 0.2
const ymaxInput = document.getElementById("ymax");
ctx.strokeStyle = "white";
ctx.lineWidth = 2;
ctx.strokeRect(0, 0, canvas.width, canvas.height);

let yMax = parseFloat(ymaxInput.value);
let max_iter = 1000

ymaxInput.addEventListener("input", (event) => {
    yMax = parseFloat(event.target.value);
});

function drawAxesAndLabels() {
    const xAxisYPosition = win0_height - 50;
    const yAxisXPosition = 50;

    ctx.beginPath();
    ctx.moveTo(yAxisXPosition, xAxisYPosition);
    ctx.lineTo(canvas.width - 50, xAxisYPosition);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(yAxisXPosition, xAxisYPosition);
    ctx.lineTo(yAxisXPosition, 50);
    ctx.stroke();

    for (let i = 0; i <= 10; i++) {
        const x = yAxisXPosition + (canvas.width - 100) * i / 10;
        ctx.fillText(Math.round(max_iter / 10 * i), x - 10, xAxisYPosition + 20);
    }

    for (let i = 0; i <= 4; i++) {
        const y = xAxisYPosition - (xAxisYPosition - 50) * i / 4;
        ctx.fillText((yMax * i / 4).toFixed(1), 10, y + 5);
    }

    
    ctx.beginPath();
    ctx.moveTo(0, win0_height);
    ctx.lineTo(canvas.width, win0_height);
    ctx.stroke();
}

async function updateLossCurve() {
    let response = await fetch("/loss_data");
    let data = await response.json();
    let loss_data = data.loss_data;

    let all_max = data.all_max
    let all_data = data.all_data
    let all_window = data.all_window

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxesAndLabels();
    
    y0 = loss_data[0].y
    py0 = win0_height - 50 - (y0 / yMax) * win0_height

    ctx.strokeStyle = 'rgb(128, 128, 128)';
    ctx.beginPath();
    ctx.moveTo(
        50,
        py0
    );

    for (let i = 1; i < loss_data.length; i++) {
        ctx.lineTo(
            50 + (i / max_iter) * (canvas.width - 100),
            win0_height - 50 - (loss_data[i].y / yMax) * win0_height
        );
    }

    ctx.stroke();

    if (loss_data.length > 0) {
        const currentIteration = loss_data[loss_data.length - 1].x;
        const currentLoss = loss_data[loss_data.length - 1].y;
        ctx.font = "16px Arial";
        ctx.fillText(`Num of logs: ${max_iter}`, canvas.width - 200, 30);
        ctx.fillText(`Iteration: ${currentIteration}`, canvas.width - 200, 50);
        ctx.fillText(`Loss: ${currentLoss.toFixed(4)}`, canvas.width - 200, 70);
    }

    ctx.beginPath()
    all_y0 = all_data[0]
 
    w2y0 = win0_height + win1_height - all_y0 / all_max * win1_height
    ctx.moveTo(0, w2y0)

    for (let j = 1; j < all_data.length; j++) {
        wjy = win0_height + win1_height - all_data[j] / all_max * win1_height
        ctx.lineTo(j,wjy)
    }
    ctx.stroke();

    ctx.font = "16px Arial";
    ctx.fillText("Loss", canvas.width / 2 - 50, win0_height + 20)

}

function updateAndRedraw() {
    updateLossCurve();
    drawAxesAndLabels();
    setTimeout(updateAndRedraw, 1000);
}

updateAndRedraw();