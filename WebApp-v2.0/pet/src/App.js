import React,{ useRef,useState } from 'react'
import { Button,Navbar,Container } from 'react-bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css'
import "./App.css"

import Preprocessing from './utils/preprocessing'
import predict from './utils/prediction'
import argmax from './utils/argmax'
import Showresult from './utils/showResult'

export default function App() {

	const CanvasRef1 = useRef();
	const CanvasRef2 = useRef();
	const [prediction,setprediction] = useState(null);

	function drawCanvas(img){
		const canvas1 = CanvasRef1.current.getContext("2d");
		const canvas2 = CanvasRef2.current.getContext("2d");
		const url = URL.createObjectURL(img);
		const image1 = new Image(224,224);
		const image2 = new Image(500,400);
		image1.src = url;
		image2.src = url;
		image1.onload = function(){
			canvas1.drawImage(image1,0,0,224,224);
		}
		image2.onload = function(){
			canvas2.drawImage(image2,0,0,500,400);
		}
	}

  return ( 
	<div >
		<Navbar bg="dark" sticky='top'>
        	<Container>
          		<Navbar.Brand id="brand-name"> Pets Classifier </Navbar.Brand>
        	</Container>
		</Navbar>

		<Container>
			<div id="app-details">
				<div id="app-name">
					<h1>Your Pet Classifier</h1>
				</div>
				<div id="app-des">
					<p>select your pet either cat or a dog this app will say your is it a cat or dog </p>
				</div>
			</div>
    	</Container>

    	<Container>
			<div id="canvas-container">
				<canvas id="canvas-image" width={224} height={224} ref={CanvasRef1}></canvas>
				<canvas width={500} height={400} ref={CanvasRef2}></canvas>
			</div>
		</Container>

		<div id="whole-container">
			<div id="user-container">
				<Container>
					<div id="result-area">
						<input
								id="file-input"
								type="file"
								name="user-image"
								onChange={(event) => {
										drawCanvas(event.target.files[0]);
								}}
							/>
					</div>
					<div className="result-btn">
						<Button id="predict-btn" onClick={async () => {
								const inputTensor = Preprocessing(CanvasRef1);
								await predict(inputTensor).then((prediction) => {
									setprediction(argmax(prediction.output.data));
								})
						}}>Predict</Button>
					</div>
				</Container>
			</div>

			<div id="ai-container">
				<Container>
					<div id="prediction-area">
						<Showresult prediction={prediction} />
					</div>
				</Container>
			</div>
		</div>

	</div>
  )
}


