# Jedimt

**Jedimt** is a powerful and extensible compiler and real-time processor designed to handle multiple programming languages. It integrates advanced features such as AI-driven extension generation, a spiral staircase-inspired threading model for efficient multi-threaded processing, real-time signal processing, comprehensive compression techniques, and a robust API interface for seamless integration into various workflows.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Compiling and Executing Code](#compiling-and-executing-code)
  - [Real-Time Operations](#real-time-operations)
- [API Reference](#api-reference)
  - [REST API Endpoints](#rest-api-endpoints)
  - [WebSocket Events](#websocket-events)
- [Examples](#examples)
- [Tutorials](#tutorials)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Language Auto-Detection:** Automatically identifies the programming language of uploaded scripts using machine learning models.
- **AI-Driven Extension Generation:** Utilizes Google's Gemini AI to generate language extensions, including syntax rules, compiler configurations, and code samples.
- **Spiral Staircase Threading Model:** Innovative threading mechanism inspired by spiral staircases for efficient and scalable multi-threaded processing.
- **Real-Time Signal Processing:** Vector-based real-time operations using `numpy` for advanced signal manipulation.
- **Advanced Compression:** Implements statistical range compression and spiral staircase vector folding compression for optimized data management.
- **Comprehensive API Interface:** RESTful APIs and WebSocket support for seamless integration with external systems and real-time communication.
- **Inspector Dashboard:** Live terminal outputs and debugging information to monitor and manage script executions.
- **Extensible Architecture:** Modular design allowing easy integration of new languages and features through extensions.

---

## Installation

### Prerequisites

- **Python 3.7+**
- **GitHub Linguist** (optional, for enhanced language detection)
  
  ```bash
  gem install github-linguist

Clone the Repository

git clone 
https://github.com/KTMO24/jedimt.git
cd jedimt

Install Dependencies

It’s recommended to use a virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Note: Ensure the following Python packages are installed:
	•	llvmlite
	•	numpy
	•	google-generativeai
	•	flask
	•	flask-socketio
	•	guesslang

You can install them using:

pip install llvmlite numpy google-generativeai flask flask-socketio guesslang

Configure Environment Variables

Obtain a Gemini API key from Google and set it as an environment variable:

export GEMINI_API_KEY=your_gemini_api_key  # On Windows: set GEMINI_API_KEY=your_gemini_api_key

Configuration

Language Definitions

Jedimt uses lang_def.json files to define the syntax, keywords, operators, and types for supported languages. When loading a new language extension, ensure the lang_def.json is correctly formatted.

Loading Extensions

Extensions are packaged as ZIP archives containing:
	•	lang_def.json: Language configuration.
	•	edge.py: Custom functions and classes for the compiler.
	•	readme.txt: License and information.
	•	examples/: Directory containing code samples.

To load an extension:

jedimt.load_extension('path_to_extension.zip')

Example:

jedimt.load_extension('extensions/Rust_extension.zip')

Usage

Compiling and Executing Code

1. Initialize Jedimt

from jedimt import Jedimt

# Initialize in compile mode with Gemini API Key
gemini_api_key = "YOUR_GEMINI_API_KEY"
jedimt = Jedimt(mode="compile", gemini_api_key=gemini_api_key)

2. Compile and Execute a Script

source_code = '''
let x: i32 = 10;
const PI: f64 = 3.1415;

fn main() {
    print("Hello, Jedimt!");
    let mut y: i32 = x + 5;
    y = y * 2;
    if y > 20 {
        print("y is greater than 20");
    } else {
        print("y is 20 or less");
    }
}
'''

script_id = "script_001"

# Compile the code
compile_result = jedimt.compile_code(source_code, script_id)
print("Compile Result:", compile_result)

# Execution is handled automatically via threading

3. Real-Time Mode

Initialize Jedimt in realtime mode:

jedimt = Jedimt(mode="realtime", gemini_api_key=gemini_api_key)

Perform real-time operations:

# Process a signal
jedimt.realtime_operation("process_signal", index=0, value=5.0)

# Apply timelock loops
jedimt.realtime_operation("apply_timelock_loops")

# Gather future insights
jedimt.realtime_operation("gather_future_insights")

# Perform pulse measurement
jedimt.realtime_operation("pulse_measurement", index=0)

API Reference

Jedimt provides both RESTful APIs and WebSocket events for interacting with the system.

REST API Endpoints

1. Upload and Compile Code
	•	Endpoint: /upload
	•	Method: POST
	•	Description: Uploads and compiles a code snippet.
	•	Payload:

{
  "code": "let x: i32 = 10;\nconst PI: f64 = 3.1415;\n\nfn main() {\n    print(\"Hello, Jedimt!\");\n    let mut y: i32 = x + 5;\n    y = y * 2;\n    if y > 20 {\n        print(\"y is greater than 20\");\n    } else {\n        print(\"y is 20 or less\");\n    }\n}",
  "script_id": "script_001"
}


	•	Response:

{
  "message": "Code uploaded and compiled successfully.",
  "language": "Rust",
  "script_id": "script_001"
}



2. Run Compiled Script
	•	Endpoint: /run
	•	Method: POST
	•	Description: Executes a previously compiled script.
	•	Payload:

{
  "script_id": "script_001"
}


	•	Response:

{
  "message": "Script 'script_001' is being executed."
}



3. Inspect Scripts
	•	Endpoint: /inspect
	•	Method: GET
	•	Description: Displays the inspector dashboard in the terminal.
	•	Response:

{
  "message": "Inspection data displayed in the terminal."
}



WebSocket Events

1. Connect
	•	Event: connect
	•	Description: Establishes a WebSocket connection.
	•	Response:

{
  "data": "Connected to Jedimt API"
}



2. Run Command
	•	Event: run_command
	•	Description: Executes a specific command within a script.
	•	Payload:

{
  "script_id": "script_001",
  "command": "restart"
}


	•	Response:

{
  "command_result": {
    "script_id": "script_001",
    "result": "Command 'restart' executed on script 'script_001'."
  }
}



3. Receive Command Results
	•	Event: command_result
	•	Description: Receives the result of a command execution.
	•	Payload:

{
  "script_id": "script_001",
  "result": "Command 'restart' executed on script 'script_001'."
}



4. Receive Errors
	•	Event: error
	•	Description: Receives error messages from the server.
	•	Payload:

{
  "message": "Error message detailing what went wrong."
}

Examples

1. Compiling a Simple Script

from jedimt import Jedimt

gemini_api_key = "YOUR_GEMINI_API_KEY"
jedimt = Jedimt(mode="compile", gemini_api_key=gemini_api_key)

source_code = '''
let x: i32 = 10;
const PI: f64 = 3.1415;

fn main() {
    print("Hello, Jedimt!");
    let mut y: i32 = x + 5;
    y = y * 2;
    if y > 20 {
        print("y is greater than 20");
    } else {
        print("y is 20 or less");
    }
}
'''

script_id = "script_001"
compile_result = jedimt.compile_code(source_code, script_id)
print("Compile Result:", compile_result)

Output:

Compile Result: {'language': 'Rust', 'script_id': 'script_001'}
Inspector Dashboard:
Script ID: script_001
Language: Rust
Input: N/A
Output: N/A
------------------------------
SpiralThreadManager: Thread script_001 assigned to slot 0.

2. Running the Compiled Script

The execution is handled automatically upon compilation. The inspector dashboard will update accordingly.

Output:

Jedimt: Executing script 'script_001'...
Script 'script_001' executed successfully.
SpiralThreadManager: Thread script_001 removed from slot 0.

Tutorials

1. Adding a New Language Extension

Step 1: Prepare the Extension Content

Ensure your extension ZIP archive contains:
	•	lang_def.json: Defines keywords, operators, types, and syntax rules.
	•	edge.py: Contains custom functions and classes for parsing and compiling.
	•	readme.txt: License and documentation.
	•	examples/: Directory with code samples.

Step 2: Load the Extension

jedimt.load_extension('extensions/NewLanguage_extension.zip')

Step 3: Compile and Execute Scripts in the New Language

source_code = '''
// Your code in the new language
'''

script_id = "script_002"
compile_result = jedimt.compile_code(source_code, script_id)
print("Compile Result:", compile_result)

2. Real-Time Signal Processing

Step 1: Initialize Jedimt in Real-Time Mode

jedimt = Jedimt(mode="realtime", gemini_api_key=gemini_api_key)

Step 2: Perform Real-Time Operations

# Process a signal
jedimt.realtime_operation("process_signal", index=0, value=5.0)

# Apply timelock loops
jedimt.realtime_operation("apply_timelock_loops")

# Gather future insights
jedimt.realtime_operation("gather_future_insights")

# Perform pulse measurement
jedimt.realtime_operation("pulse_measurement", index=0)

Output:

RealTimeProcessor: Pulse Measurement at index 0: 0.2
Inspector Dashboard:
Script ID: script_001
Language: Rust
Input: N/A
Output: Script 'script_001' executed successfully.
------------------------------

Contributing

We welcome contributions to Jedimt! Whether you’re reporting bugs, suggesting features, or submitting pull requests, your input is valuable.

How to Contribute
	1.	Fork the Repository
Click the “Fork” button at the top-right of the repository page to create a copy of the repository under your GitHub account.
	2.	Clone the Forked Repository

git clone https://github.com/yourusername/jedimt.git
cd jedimt


	3.	Create a New Branch

git checkout -b feature/YourFeatureName


	4.	Make Changes and Commit

git commit -m "Add your commit message here"


	5.	Push to the Branch

git push origin feature/YourFeatureName


	6.	Open a Pull Request
Navigate to the original repository and click “Compare & pull request” to submit your changes for review.

Coding Standards
	•	Follow PEP 8 guidelines for Python code.
	•	Include docstrings for all classes and methods.
	•	Write meaningful commit messages.

Reporting Issues

If you encounter any bugs or have feature requests, please open an issue in the Issues section.

License

This project is licensed under the BSD 3-Clause License. See the LICENSE file for details.

Contact

For any inquiries or support, please contact:
	•	Author: Travis Michael O’Dell
	•	Email: travismichaelodell@tutamail.com
	•	

Acknowledgements
	•	Google’s Gemini AI: For providing powerful generative capabilities.
	•	OpenAI: For the foundational inspiration and AI models.
	•	Community Contributors: For their valuable feedback and contributions.

