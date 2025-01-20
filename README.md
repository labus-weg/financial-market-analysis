## 🚀 Financial Market Anomaly Detection

This project is an intermediate market anomaly detection tool built to help investors, traders, and data scientists identify unusual behavior in financial markets. The tool uses basic/classical machine learning algorithms to analyze market data for any given dataset by detecting outliers, providing visualisation and AI support to gain insights.

---

## 🎯 Objective
As introduced in the [Headstarter AI Accelerator Program](https://app.headstarter.co/), the goal of this project is to build a comprehensive anomaly detection system for financial market analysis that can:
- Detect unusual market behaviors or "anomalies."
- Visualize the detected anomalies on time series graphs.
- Suggest investment strategies based on detected anomalies.
- Provide AI-driven insights through a chatbot advisor.

---

## 🛠️ Features

### 1. **Anomaly Detection Models**
   - **Isolation Forest**
   - **One-Class SVM**
   - **Local Outlier Factor (LOF)**

These models are used to detect anomalies in the financial data, such as price spikes, drastic drops, and other irregularities.

### 2. **Visualization**
   - **Anomaly Timeline Plot**: Visualizes detected anomalies over time.
   - **Anomaly Score Plot**: Shows the anomaly scores assigned by the model, giving insights into how likely a data point is an anomaly.

### 3. **Investment Strategy Suggestions**
   Based on the detected anomalies, the system can be optimised to suggest potential investment strategies (buy, sell, hold) to guide users' decisions. Currently, this is not entirely attained but is being considered in its future iteration.

### 4. **AI Financial Advisor Chatbot**
   - Offers personalized financial advice based on user input.
   - Powered by OpenAI's GPT model.
   - Needs further debugging to generate accurate responses.

---

## 📊 Data Input

You can upload your market data in CSV or Excel formats. The tool supports a variety of financial datasets, including stock prices, market indices, and trading volumes.

---

## 💻 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/labus-weg/financial-market-anomaly-detection.git
   cd market-anomaly-detection
   ```

2. **Install the required dependencies:**
   You need Python 3.x to run the project. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   Launch the application using Streamlit:
   ```bash
   streamlit run app.py
   ```

4. **Upload your market data:**
   Use the sidebar to upload your dataset in CSV or Excel format. Adjust the sensitivity slider to control the anomaly detection sensitivity.

---

## 🧠 How It Works

1. **Data Preprocessing**: 
   - Missing values are automatically handled by filling them with the column mean.
   - Numerical data is standardized using `StandardScaler`.

2. **Anomaly Detection**:
   - The model (Isolation Forest, One-Class SVM, or LOF) is trained using the uploaded market data.
   - Anomalies are predicted and visualized on a timeline plot.

3. **Investment Strategy Suggestions**:
   - Based on the detected anomalies, the system provides investment strategies such as buying, selling, or holding.

4. **AI Chatbot**:
   - A chatbot interface is provided to give AI-driven investment advice.

---

## 🛠️ Libraries & Tools Used

- **Streamlit**: Web framework for interactive data apps.
- **Scikit-Learn**: Machine learning models for anomaly detection.
- **Plotly**: Interactive plots for data visualization.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **OpenAI GPT**: For AI-driven financial advice.

---

## 📌 Example Use Case

1. **Upload Data**: Upload historical market data (e.g., stock prices, market indices).
2. **Adjust Sensitivity**: Use the slider to adjust the sensitivity of the anomaly detection model.
3. **View Results**: View the detected anomalies on a timeline plot and examine the anomaly scores.
4. **Investment Strategy**: Based on the detected anomalies, the system will suggest buy, sell, or hold strategies.
5. **Chat with AI Advisor**: Ask your financial questions, and the chatbot will provide personalized advice.

---

## ⚙️ Customization

- You can modify the sensitivity settings and experiment with different anomaly detection models to suit your specific needs.
- The code supports CSV and Excel formats for easy integration with various financial datasets.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤖 Chatbot Advisor

Need financial advice? Ask our AI-powered chatbot! Simply type your question in the input box and get insights powered by GPT. It can help you understand market trends, anomalies, and guide you in making informed decisions.

---

## 👥 Contributing

If you'd like to contribute to this project, feel free to fork the repository, make improvements, and submit a pull request. Whether it's fixing bugs, adding features, or improving documentation, your contributions are welcome!

---

## 🔧 Troubleshooting

- **Issue**: "st is not defined"
  - **Solution**: Ensure you are importing Streamlit correctly by adding `import streamlit as st` at the top of your `app.py`.

- **Issue**: "Unsupported file type"
  - **Solution**: Make sure you are uploading either a CSV or Excel file.

---

## 📅 Future Enhancements

- Add more anomaly detection models like **Autoencoders**.
- Integrate real-time data feeds for live anomaly detection.
- Implement more advanced investment strategies based on detected anomalies.
- Enhance the AI chatbot with more market-specific insights.

---

## 👨‍💻 Author

- **Nafisa Nawrin Labonno**
- GitHub: [labus-weg]([https://github.com/your-username](https://github.com/labus-weg/)

---
## ©️ Credits
Special thanks to Angelica Iacovelli for introducing the challenge and to Faizan Ahmed & Yasin Ehsan at Headstarter AI (Accelerator Program) for their continuous support.

## 💬 Feedback & Support

If you have any questions or feedback, feel free to reach out or open an issue on GitHub. I'm always happy to help!

```
