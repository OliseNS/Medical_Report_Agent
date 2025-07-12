# DCC Medical Report Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI-powered medical report analyzer that uses large language models via OpenRouter to provide comprehensive health insights, risk assessments, and medical recommendations. This tool is designed with a focus on kidney-related conditions, providing detailed analysis for nephrology-related reports.
ed to replace this with a real screenshot of your app -->

## ✨ Key Features

- **🤖 AI-Powered Analysis**: Leverages the DeepSeek Chat v3.1 model for deep analysis.
- **🩺 Kidney Disease Focus**: Specialized prompts and logic for analyzing reports related to kidney disease, dialysis, and transplantation.
- **📊 Comprehensive Health Insights**: Generates detailed insights on medical conditions, lab values, and risk factors.
- **⚠️ Risk Assessment**: Automated risk scoring and urgent alert detection for critical findings.
- **💊 Medication & Allergy Tracking**: Extracts and lists medications and allergies from reports.
- **📈 Dynamic Health Scoring**: Calculates an overall health score (1-100) based on the report's contents.
- **🌐 Interactive Web Interface**: A user-friendly web application built with Streamlit.
- **📋 Export Options**: Allows exporting analysis results as JSON or CSV files.

## 🚀 Getting Started

Follow these instructions to set up and run the DCC Medical Report Agent on your local machine.

### Prerequisites

- Python 3.8+
- An [OpenRouter API key](https://openrouter.ai/)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/OliseNS/Medical_Report_Agent.git
    cd Medical_Report_Agent
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The application requires an OpenRouter API key to function. You can set this in one of two ways:

1.  **Environment Variable (Recommended):**
    Set an environment variable named `OPENROUTER_API_KEY`.
    ```bash
    export OPENROUTER_API_KEY="your-api-key-here"
    ```

2.  **Streamlit Secrets (for Deployment):**
    If you are deploying to Streamlit Community Cloud, use their secrets management. Add your API key to your `secrets.toml` file.
    ```toml
    OPENROUTER_API_KEY = "your-api-key-here"
    ```

### Running the Application

Once the setup is complete, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## 💻 How to Use

1.  **AI Model**: The system uses the DeepSeek Chat v3.1 model for analysis.
2.  **Choose an Input Method**:
    - **📝 Text Input**: Paste the medical report text directly into the text area.
    - **📁 File Upload**: Upload a `.txt` file containing the medical report.
    - **📋 Sample Reports**: Select one of the provided sample reports to see the agent in action.
3.  **Analyze**: Click the "Analyze Report" button to start the analysis.
4.  **View Results**: The results will be displayed in different tabs:
    - **Overview**: A summary of the patient, key metrics, and urgent alerts.
    - **Health Insights**: Detailed breakdown of identified conditions, their severity, and recommendations.
    - **Risk Assessment**: A visual representation of the risk level and contributing factors.
    - **Detailed Report**: The complete analysis in a structured format.

## 🛡️ Disclaimer

⚠️ **Important:** This AI tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns or before making any health decisions.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 