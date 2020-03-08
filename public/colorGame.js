var numSquares = 10;
var colors = [];
var pickedColor;
var squares = document.querySelectorAll(".square");
var colorDisplay = document.getElementById("colorDisplay");
var messageDisplay = document.querySelector("#message");
var h1 = document.querySelector("h1");
var resetButton = document.querySelector("#reset");
var modeButtons = document.querySelectorAll(".mode");
var questions = [
	{
		question:"I will never succeed in life",
		answer:"1"
	},
	{
		question:"Everyone in my family is rude",
		answer:"2"
	},
	{
		question:"I am going to die bankrupt",
		answer:"3"
	}
]
var i=0;
var j=1;
function getSquares(id)
{		
		if(id == i+1){
			document.getElementById("message").innerHTML="Correct"
			document.getElementById("btn").style.visibility="visible"
			
		}else{
			document.getElementById("message").innerHTML="Incorrect ,Correct answer is "+ (i+1)
			document.getElementById("btn").style.visibility="hidden"

		}
	
}

function nextquestion()
{
	if(j==3)
	{
		document.getElementById("message").innerHTML="Thank you";
		document.getElementById("link").style.visibility="visible";
		document.getElementById("btn").style.visibility="hidden";

	}
	else
	{
		document.getElementById("question").innerHTML=questions[j].question;
		document.getElementById("message").innerHTML="";
		document.getElementById("btn").style.visibility="hidden"
		i++;j++;
	}
}



