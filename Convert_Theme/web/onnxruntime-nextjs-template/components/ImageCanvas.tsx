import { useRef, useState } from "react";
import { inferenceSqueezenet } from "../utils/predict";
import styles from "../styles/Home.module.css";

interface Props {
  height: number;
  width: number;
  className: string;
}

const ImageCanvas = (props: Props) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasTargetRef = useRef<HTMLCanvasElement>(null);
  const canvasOutputRef = useRef<HTMLCanvasElement>(null);
  var image: HTMLImageElement;
  var imageTarget: HTMLImageElement;
  var imageOutput: HTMLImageElement;

  const [topResultLabel, setLabel] = useState("");
  const [topResultConfidence, setConfidence] = useState("");
  const [inferenceTime, setInferenceTime] = useState("");

  // Load the image from the IMAGE_URLS array
  const getImage = () => {
    const LOWER = 26;
    const UPPER = 32;
    const random = Math.floor(Math.random() * (UPPER - LOWER) + LOWER);
    console.log({
      text: "Image id " + random,
      valueInput: "./light/" + random + ".png",
      valueOutput: "./dark/" + random + ".png",
    });
    return {
      text: "Image id " + random,
      valueInput: "./light/" + random + ".png",
      valueOutput: "./dark/" + random + ".png",
    };
  };

  // Draw image and other  UI elements then run inference
  const displayImageAndRunInference = () => {
    // Get the image
    image = new Image();
    imageTarget = new Image();
    var sampleImage = getImage();
    image.src = sampleImage.valueInput;
    imageTarget.src = sampleImage.valueOutput;

    // Clear out previous values.
    setLabel(`Inferencing...`);
    setConfidence("");
    setInferenceTime("");

    // Draw the image on the canvas
    const canvas = canvasRef.current;
    const ctxInput = canvas!.getContext("2d");
    console.log(props.width)
    console.log(props.height)
    image.onload = () => {
      ctxInput!.drawImage(image, 0, 0, props.width, props.height);
    };

    // Draw the image on the canvas
    const canvasOutput = canvasTargetRef.current;
    const ctxOutput = canvasOutput!.getContext("2d");
    imageTarget.onload = () => {
      ctxOutput!.drawImage(imageTarget, 0, 0, props.width, props.height);
    };

    // Run the inference
    submitInference();
  };

  const submitInference = async () => {
    // Get the image data from the canvas and submit inference.
    console.log(image.src)
    var [output, inferenceTime] = await inferenceSqueezenet(image.src);

    // Get the highest confidence.
    // var topResult = inferenceResult[0];

    // Update the label and confidence
    // setLabel(topResult.name.toUpperCase());
    // setConfidence(topResult.probability);
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);

    var width = 256,
      height = 256,
      buffer = new Uint8ClampedArray(width * height * 4); // have enough bytes
    var R = output.data.slice(0, 65536);
    var G = output.data.slice(65536, 65536 * 2);
    var B = output.data.slice(65536 * 2, 65536 * 3);

    for (var y = 0; y < height * width; y++) {
      buffer[y * 4] = R[y] * 255; // some R value [0, 255]
      buffer[y * 4 + 1] = G[y] * 255; // some G value
      buffer[y * 4 + 2] = B[y] * 255; // some B value
      buffer[y * 4 + 3] = 255; // set alpha channel
    }
    // create off-screen canvas element

    var canvas = canvasOutputRef.current;
    var ctx = canvas!.getContext("2d");

    // create imageData object
    var idata = ctx!.createImageData(width, height);

    // set our buffer as source
    idata.data.set(buffer);

    // update canvas with new data
    ctx!.putImageData(idata, 0, 0);

    // Append image to body base64 encoded
    // // create a new img object
    // imageOutput = new Image();

    // // set the img.src to the canvas data url
    // imageOutput.src = canvas!.toDataURL();

    // // append the new img object to the page
    // document.body.appendChild(imageOutput);
    // console.log(buffer);
  };

  return (
    <>
      <button className={styles.grid} onClick={displayImageAndRunInference}>
        Run inference from a random pic
      </button>
      <br />
      <div style={{ display: "flex" }}>
        <div>
          <h3>Input</h3>
          <canvas ref={canvasRef} width={props.width} height={props.height} />
        </div>
        <div>
          <h3>Target</h3>
          <canvas
            ref={canvasTargetRef}
            width={props.width}
            height={props.height}
          />
        </div>
        <div>
          <h3>Model Output</h3>
          <canvas
            ref={canvasOutputRef}
            width={props.width}
            height={props.height}
          />
        </div>
      </div>
      <span>
        {topResultLabel} {topResultConfidence}
      </span>
      <span>{inferenceTime}</span>
    </>
  );
};

export default ImageCanvas;
