const generateButton = document.getElementById('generate-button');
const inputTextBox = document.getElementById('input-text');
const outputTextBox = document.getElementById('output-text');

generateButton.addEventListener('click', () => {
	const inputText = inputTextBox.value;
	
	if (inputText) {
		const xhr = new XMLHttpRequest();
		xhr.open('POST', '/generate');
		xhr.setRequestHeader('Content-Type', 'application/json');
		xhr.onload = () => {
			if (xhr.status === 200) {
				const response = JSON.parse(xhr.responseText);
				outputTextBox.value = response.generated_text;
			} else {
				console.error(xhr.statusText);
			}
		};
		xhr.send(JSON.stringify({input_text: inputText}));
	} else {
		alert('Please enter some text');
	}
});
