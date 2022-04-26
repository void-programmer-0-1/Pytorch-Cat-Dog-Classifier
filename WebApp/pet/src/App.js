import React,{ useRef,useState } from 'react'
import { Button,Navbar,Container } from 'react-bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css'
import "./App.css"

import Preprocessing from './utils/preprocessing'
import predict from './utils/prediction'
import argmax from './utils/argmax'
import Showresult from './utils/showResult'

export default function App() {

	const CanvasRef = useRef();
	const [prediction,setprediction] = useState(null);

	function drawCanvas(img){
		const canvas = CanvasRef.current.getContext("2d");
		const url = URL.createObjectURL(img);
		const image = new Image(224,224);
		image.src = url;
		image.onload = function(){
			canvas.drawImage(image,0,0,224,224);
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
				<canvas width={224} height={224} ref={CanvasRef}></canvas>
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
								const inputTensor = Preprocessing(CanvasRef);
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


