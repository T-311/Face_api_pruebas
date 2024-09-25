// Reconocimiento facial
const run = async () => {
    const img1 = document.getElementById('image1').value;
    const img2 = document.getElementById('image2').value;

    if (!img1 || !img2) {
        alert('Por favor, ingresa ambas URLs de imágenes.');
        return;
    }

    // Mostrar las imágenes 1 y 2 en el HTML
    const faceImg1 = document.getElementById('face1');
    const faceImg2 = document.getElementById('face2');
    
    faceImg1.src = img1;  // Se actualiza la imagen 1 con la URL ingresada
    faceImg2.src = img2;  // Se actualiza la imagen 2 con la URL ingresada

    // Capturar el tiempo de inicio
    const startTime = Date.now();

    // Esperar a que ambas imágenes se carguen antes de proceder
    Promise.all([faceImg1.decode(), faceImg2.decode()]).then(async () => {
        // Cargar los modelos necesarios
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
            faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
            faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
            faceapi.nets.ageGenderNet.loadFromUri('./models'),
        ]);

        // Cargar las imágenes ingresadas
        const refFace = await faceapi.fetchImage(img1);
        const facesToCheck = await faceapi.fetchImage(img2);

        // Detectar rostros y obtener datos AI
        let refFaceAiData = await faceapi.detectAllFaces(refFace).withFaceLandmarks().withFaceDescriptors();
        let facesToCheckAiData = await faceapi.detectAllFaces(facesToCheck).withFaceLandmarks().withFaceDescriptors();

        const canvas = document.getElementById('canvas');
        faceapi.matchDimensions(canvas, facesToCheck);

        // Crear el face matcher con los datos de referencia
        let faceMatcher = new faceapi.FaceMatcher(refFaceAiData);
        facesToCheckAiData = faceapi.resizeResults(facesToCheckAiData, facesToCheck);

        // Comparar las caras y dibujar resultado
        facesToCheckAiData.forEach(face => {
            const { detection, descriptor } = face;
            let label = faceMatcher.findBestMatch(descriptor).toString();
            console.log(label);

            if (label.includes("unknown")) {
                return;
            }

            let options = { label: "Cara Comparada" };
            const drawBox = new faceapi.draw.DrawBox(detection.box, options);
            drawBox.draw(canvas);
        });

        // Capturar el tiempo de finalización
        const endTime = Date.now();
        const timeTaken = (endTime - startTime) / 1000; // Convertir a segundos

        // Mostrar el tiempo tomado en el HTML
        document.getElementById('timeTaken').textContent = `Tiempo tomado: ${timeTaken} segundos`;
    }).catch(err => {
        console.error('Error al cargar las imágenes:', err);
    });
};
