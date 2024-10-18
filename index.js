const cv = require('opencv4nodejs');
const { createWorker } = require('tesseract.js');
const moment = require('moment');

async function preprocessImage(imagePath) {
  const image = await cv.imreadAsync(imagePath);
  const gray = image.cvtColor(cv.COLOR_BGR2GRAY);
  const thresh = gray.threshold(0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
  return thresh;
}

async function extractText(image) {
  const worker = createWorker();
  await worker.load();
  await worker.loadLanguage('eng');
  await worker.initialize('eng');
  const { data: { text } } = await worker.recognize(image);
  await worker.terminate();
  return text;
}

function checkExpiryDate(text) {
  const datePattern = /\d{2}[-/]\d{2}[-/]\d{4}/;
  const dates = text.match(datePattern);
  if (dates) {
    const expiryDate = moment(dates[0], 'DD-MM-YYYY');
    if (expiryDate.isValid()) {
      return expiryDate.isAfter(moment()) ? "Product is not expired" : "Product is expired";
    }
  }
  return "Could not parse expiry date";
}

function countObjects(image) {
  const edges = image.canny(50, 150);
  const contours = edges.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  return contours.length;
}

async function main() {
  try {
    const imagePath = 'path/to/your/image.jpg';
    const preprocessedImage = await preprocessImage(imagePath);
    
    const text = await extractText(preprocessedImage);
    console.log("Extracted Text:", text);
    
    const expiryStatus = checkExpiryDate(text);
    console.log("Expiry Status:", expiryStatus);
    
    const objectCount = countObjects(preprocessedImage);
    console.log("Object Count:", objectCount);
    
    // Note: Freshness assessment would require more complex implementation
    // and possibly a pre-trained model, which is beyond the scope of this example.
    console.log("Freshness assessment not implemented in this basic example.");
    
  } catch (error) {
    console.error("An error occurred:", error);
  }
}

main();
