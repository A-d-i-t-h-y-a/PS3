// import axios from "../node_modules/axios/dist/axios.js"


async function submitForm() {
    // Serialize form data
    var formData = new FormData(document.getElementById('myForm'));
    var btn = document.getElementsByClassName("btn")[0];
    btn.disabled = true;
    // var jsonData = {};
    // formData.forEach((value, key) => {
    //     jsonData[key] = value;
    // });
    const spinner = document.getElementById("spinner");
    spinner.classList.remove("d-none");
    const loadImageBase64 = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = (error) => reject(error);
        });
    }
    
    const fileInput = document.getElementById('fileInput');

    // Check if a file is selected
    if (fileInput.files.length > 0) {
        // Pass the selected file to loadImageBase64 function
        const image = await loadImageBase64(fileInput.files[0]);
        const tbody = document.getElementsByTagName("tbody")[0];
        axios({
            method: "POST",
            url: "/submit",
            data: { image },
            headers: {
                "Content-Type": "application/json",
            },
        })
        .then(function(response) {
            // console.log("data received");
            // // Update the resultContainer with the received data
            console.log(response.data)
            if(response.data["roboflow_result"]){
                var resultImage = document.getElementById('resultImage');
                spinner.classList.add("d-none");
                resultImage.src = 'data:image/jpeg;base64,'+response.data.roboflow_result;
                resultImage.setAttribute("width", "75%")
                resultImage.setAttribute("height", "75%")
                // var resultContainer = document.getElementById('resultContainer')
                // resultContainer.innerHTML = `
                //     <h2>Results</h2>
                //     <p>Roboflow Result: ${JSON.stringify(response.data)}</p>
                // `;
                btn.disabled = false;
                let temp = ["aadharno", "details", "qr", "emblem", "goi", "image"];
                let t = response.data["detected"];
                tbody.innerHTML = `<tr>
                <th>Feature</th>
                <th>Image</th>
                <th>detected(✔️|❌)</th>
                <th>extracted</th>
            </tr>`
                for(let i of temp){
                    const tr = document.createElement("tr");
                    let det = i in t;
                    tr.innerHTML = `<td scope="row" class="feature">${i}</td>
                    <td class="img">${det ? `<img src="/static/detected/${t[i]}.jpg" alt="">` : 'Unable to Detect'}</td>
                    <td class="verify">${det ? '✔️' : '❌'}</td>
                    <td class="extract">N/A</td>`
                    tbody.appendChild(tr);
                }
            }
        })
        .catch(function(error) {
            console.error('Error:', error);
            // Handle errors if needed
        });
    } else {
        console.error('No file selected');
        // Handle the case where no file is selected
    }
}
