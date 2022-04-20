import { Button,Badge } from 'react-bootstrap'
import React from 'react'

export default function Showresult(props){
    const prediction = props.prediction;
   
    if(prediction == null){
        return(
            <div>
                 <Button variant="dark">
                     <Badge bg="secondary">Select image and click predict</Badge>
                </Button>
            </div>
        )
    }

    if(prediction != null){
        return(
            <div>
                    <Button variant="dark">
                        Predicted a <Badge bg="secondary">{ prediction }</Badge>
                    </Button>
            </div>
        )
    }
   
}