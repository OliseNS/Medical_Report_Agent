import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from medical_agent import MedicalReportAnalyzer, MedicalReportAnalysis
from typing import Dict, Any
import base64

# Page configuration
st.set_page_config(
    page_title="DCC Medical Report Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for DCC theme (black, white, blue, red)
st.markdown("""
<style>
    /* Global theme colors */
    :root {
        --dcc-black: #212121; /* Softer black for text */
        --dcc-white: #ffffff;
        --dcc-blue: #007bff; /* Brighter blue */
        --dcc-red: #dc3545; /* Brighter red */
        --dcc-light-blue: #e7f3ff;
        --dcc-dark-blue: #004085;
        --dcc-gray: #495057;
        --dcc-light-gray: #f8f9fa;
        --dcc-success: #28a745;
        --dcc-warning: #ffc107;
    }
    
    /* Main app background and text color */
    body, .main, .block-container {
        background-color: var(--dcc-light-gray);
        color: var(--dcc-black);
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--dcc-dark-blue);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .tagline {
        text-align: center;
        color: var(--dcc-gray);
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }

    .sub-header {
        color: var(--dcc-dark-blue);
        font-weight: bold;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--dcc-light-blue);
        padding-bottom: 0.5rem;
    }
    
    /* Health score styling */
    .health-score {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: linear-gradient(135deg, var(--dcc-blue), var(--dcc-dark-blue));
        color: var(--dcc-white); /* White text on dark background */
        border: 3px solid var(--dcc-blue);
    }
    .health-score * {
        color: var(--dcc-white); /* Ensure child elements also have white text */
    }
    
    /* Alert and insight boxes */
    .alert-box {
        background-color: #f8d7da; /* Lighter red background */
        border-left: 4px solid var(--dcc-red);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #721c24; /* Darker red text */
    }
    .alert-box * {
        color: #721c24;
    }
    
    .insight-box {
        background-color: var(--dcc-white);
        border: 1px solid #dee2e6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        border-left: 4px solid var(--dcc-blue);
    }
    .insight-box h4 {
        color: var(--dcc-dark-blue);
        margin-top: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--dcc-dark-blue);
    }
    .css-1d391kg, .css-1d391kg * {
        color: var(--dcc-white);
    }
    .css-1d391kg h2 {
        color: var(--dcc-white);
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--dcc-blue) !important;
        color: var(--dcc-white) !important; 
        border: 1px solid var(--dcc-blue) !important;
        border-radius: 5px;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: var(--dcc-dark-blue) !important;
        border-color: var(--dcc-dark-blue) !important;
    }

    /* Input elements styling */
    .stTextArea textarea, .stTextInput input, div[data-baseweb="select"] > div {
        background-color: var(--dcc-white) !important;
        color: var(--dcc-black) !important;
        border: 1px solid #ced4da !important;
        border-radius: 5px;
    }
    .stTextArea textarea:focus, .stTextInput input:focus, div[data-baseweb="select"] > div:focus-within {
        border-color: var(--dcc-blue) !important;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom-color: #dee2e6;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--dcc-blue);
        border-bottom: 3px solid var(--dcc-blue);
    }
    
    /* Logo styling */
    .logo-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .logo-container img {
        max-width: 200px;
        height: auto;
    }
    
    /* Risk level indicators */
    .risk-low {
        background-color: var(--dcc-success);
        color: var(--dcc-white);
    }
    
    .risk-medium {
        background-color: var(--dcc-warning);
        color: var(--dcc-black); /* Black text on yellow for visibility */
    }
    
    .risk-high {
        background-color: var(--dcc-red);
        color: var(--dcc-white);
    }
    
    /* Footer styling */
    .footer {
        background-color: var(--dcc-white);
        color: var(--dcc-gray);
        padding: 1.5rem;
        text-align: center;
        border-top: 1px solid #dee2e6;
    }
    .footer p, .footer strong {
        color: var(--dcc-gray);
    }
    
    /* General text elements */
    p, li, strong {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Streamlit-specific overrides for better visibility */
    .stMetric {
        background-color: var(--dcc-white);
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
    }
    .stMetric label {
        color: var(--dcc-gray);
    }
    
    /* === NEW RULES FOR VISIBILITY === */
    /* I am removing the previous global override and adding more specific ones. */

    /* Set a base color for all text, which can be overridden. */
    .main .block-container {
        color: var(--dcc-black);
    }
    
    /* Target specific components that are still not visible. */
    
    /* 1. Key Metrics */
    .stMetric [data-testid="stMetricLabel"], .stMetric [data-testid="stMetricValue"] {
        color: var(--dcc-black) !important;
    }
    
    /* 2. Tabs */
    .stTabs [data-baseweb="tab"] {
        color: var(--dcc-black) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--dcc-blue) !important;
        border-bottom: 3px solid var(--dcc-blue);
    }
    
    /* 3. Disabled Text Area for Sample Reports */
    .stTextArea textarea:disabled {
        background-color: #e9ecef !important;
        color: #000000 !important;
    }

</style>
""", unsafe_allow_html=True)

# DCC Logo URL
DCC_LOGO_URL = "https://eadn-wc01-6859330.nxedge.io/wp-content/uploads/2021/07/DCC-Logo-Clause_Rebranded_888404-600x227.png"

@st.cache_resource
def load_analyzer():
    """Load the medical report analyzer"""
    return MedicalReportAnalyzer()

def create_health_score_gauge(score: int) -> go.Figure:
    """Create a gauge chart for health score with DCC theme"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Health Score", 'font': {'color': '#004085', 'size': 20}},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#212121'},
            'bar': {'color': "#007bff"},
            'steps': [
                {'range': [0, 40], 'color': "#f8d7da"},
                {'range': [40, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#d4edda"}
            ],
            'threshold': {
                'line': {'color': "#dc3545", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_risk_assessment_chart(risk_data: Dict[str, Any]) -> go.Figure:
    """Create a risk assessment chart with DCC theme"""
    risk_levels = ['Low', 'Medium', 'High']
    risk_colors = ['#28a745', '#ffc107', '#dc3545']
    
    # Create risk level distribution
    fig = go.Figure(data=[
        go.Bar(
            x=[risk_data['risk_level']],
            y=[risk_data['risk_score']],
            marker_color=risk_colors[risk_levels.index(risk_data['risk_level'])],
            text=f"Risk Score: {risk_data['risk_score']}",
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Risk Assessment",
        xaxis_title="Risk Level",
        yaxis_title="Risk Score",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#212121'}
    )
    return fig

def display_analysis_results(analysis: MedicalReportAnalysis):
    """Display comprehensive analysis results with DCC theme"""
    
    # Header with health score
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="health-score">
            Health Score: {analysis.overall_health_score}/100
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Health Insights", "📈 Risk Assessment", "📋 Detailed Report"])
    
    with tab1:
        # Overview tab
        col1, col2 = st.columns(2)
        
        with col1:
            # Health score gauge
            fig_gauge = create_health_score_gauge(analysis.overall_health_score)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Key metrics
            st.markdown('<h3 class="sub-header">📈 Key Metrics</h3>', unsafe_allow_html=True)
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("Conditions Identified", len(analysis.health_insights))
                st.metric("Abnormal Values", len(analysis.abnormal_values))
            with col1_2:
                st.metric("Medications", len(analysis.medications))
                st.metric("Allergies", len(analysis.allergies))
        
        with col2:
            # Patient summary
            st.markdown('<h3 class="sub-header">👤 Patient Summary</h3>', unsafe_allow_html=True)
            st.write(analysis.patient_summary)
            
            # Key findings
            st.markdown('<h3 class="sub-header">🔍 Key Findings</h3>', unsafe_allow_html=True)
            for finding in analysis.key_findings:
                st.write(f"• {finding}")
            
            # Urgent alerts
            if analysis.urgent_alerts:
                st.markdown('<h3 class="sub-header">⚠️ Urgent Alerts</h3>', unsafe_allow_html=True)
                for alert in analysis.urgent_alerts:
                    st.markdown(f"""
                    <div class="alert-box">
                        {alert}
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        # Health insights tab
        st.markdown('<h3 class="sub-header">🏥 Health Insights</h3>', unsafe_allow_html=True)
        
        for i, insight in enumerate(analysis.health_insights):
            severity_color = {
                'Low': '#28a745',
                'Medium': '#ffc107',
                'High': '#dc3545',
                'Critical': '#dc3545'
            }.get(insight.severity, '#007bff')
            
            st.markdown(f"""
            <div class="insight-box" style="border-left-color: {severity_color}">
                <h4>{insight.condition} (Severity: {insight.severity})</h4>
                <p><strong>Description:</strong> {insight.description}</p>
                <p><strong>Risk Factors:</strong> {', '.join(insight.risk_factors)}</p>
                <p><strong>Follow-up:</strong> {insight.follow_up}</p>
                <p><strong>Recommendations:</strong></p>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in insight.recommendations])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        # Risk assessment tab
        st.markdown('<h3 class="sub-header">⚠️ Risk Assessment</h3>', unsafe_allow_html=True)
        
        analyzer = load_analyzer()
        risk_data = analyzer.get_risk_assessment(analysis)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk chart
            fig_risk = create_risk_assessment_chart(risk_data)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            # Risk factors
            st.markdown('<h4 style="color: #004085;">Risk Factors</h4>', unsafe_allow_html=True)
            for factor in risk_data['risk_factors']:
                st.write(f"• {factor}")
            
            # Risk level indicator
            risk_color_map = {
                'Low': '#28a745',
                'Medium': '#ffc107',
                'High': '#dc3545'
            }
            risk_color = risk_color_map.get(risk_data['risk_level'], '#007bff')
            text_color = 'white' if risk_data['risk_level'] != 'Medium' else 'black'

            st.markdown(f"""
            <div style="background-color: {risk_color}; color: {text_color}; padding: 1rem; border-radius: 10px; text-align: center; margin-top: 1rem;">
                <h3>Risk Level: {risk_data['risk_level']}</h3>
                <p>Risk Score: {risk_data['risk_score']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        # Detailed report tab
        st.markdown('<h3 class="sub-header">📋 Detailed Analysis Report</h3>', unsafe_allow_html=True)
        
        # Generate full summary
        analyzer = load_analyzer()
        full_summary = analyzer.generate_health_summary(analysis)
        st.markdown(full_summary)
        
        # Export options
        st.markdown('<h3 class="sub-header">📤 Export Options</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export as JSON"):
                analysis_dict = analysis.dict()
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(analysis_dict, indent=2),
                    file_name=f"dcc_medical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export as CSV"):
                # Create CSV data
                data = {
                    'Category': ['Health Score', 'Conditions', 'Medications', 'Allergies'],
                    'Count': [
                        analysis.overall_health_score,
                        len(analysis.health_insights),
                        len(analysis.medications),
                        len(analysis.allergies)
                    ]
                }
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"dcc_medical_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def main():
    """Main application function"""
    
    # DCC Logo and Header
    st.markdown(f"""
    <div class="logo-container">
        <img src="{DCC_LOGO_URL}" alt="DCC Logo">
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">DCC Medical Report Agent</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="tagline">Advanced AI-powered medical report analysis with comprehensive health insights</h3>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<h2>🔧 Settings</h2>', unsafe_allow_html=True)
    
    # API Key check
    if not os.getenv("OPENROUTER_API_KEY"):
        st.sidebar.error("⚠️ OpenRouter API key not found!")
        st.sidebar.info("Please set your OPENROUTER_API_KEY environment variable")
        return
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select AI Model",
        ["openai/gpt-4", "openai/gpt-3.5-turbo", "anthropic/claude-3-opus", "anthropic/claude-3-sonnet", "meta-llama/llama-3.1-70b-instruct"],
        help="Choose the AI model for analysis"
    )
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Input Method",
        ["📝 Text Input", "📁 File Upload", "📋 Sample Reports"]
    )
    
    # Main content area
    if input_method == "📝 Text Input":
        st.markdown('<h2 class="sub-header">📝 Enter Medical Report</h2>', unsafe_allow_html=True)
        
        medical_text = st.text_area(
            "Paste your medical report here:",
            height=300,
            placeholder="Enter medical report text here..."
        )
        
        if st.button("🔍 Analyze Report", type="primary"):
            if medical_text.strip():
                with st.spinner("Analyzing medical report..."):
                    try:
                        analyzer = MedicalReportAnalyzer(model_name=model_name)
                        analysis = analyzer.analyze_report(medical_text)
                        display_analysis_results(analysis)
                    except Exception as e:
                        st.error(f"Error analyzing report: {str(e)}")
            else:
                st.warning("Please enter medical report text to analyze.")
    
    elif input_method == "📁 File Upload":
        st.markdown('<h2 class="sub-header">📁 Upload Medical Report</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a medical report file",
            type=['txt', 'pdf', 'docx'],
            help="Upload medical report files (text, PDF, or Word documents)"
        )
        
        if uploaded_file is not None:
            # For now, we'll handle text files
            if uploaded_file.type == "text/plain":
                medical_text = uploaded_file.read().decode("utf-8")
                st.text_area("File Content:", medical_text, height=200)
                
                if st.button("🔍 Analyze Uploaded Report", type="primary"):
                    with st.spinner("Analyzing uploaded report..."):
                        try:
                            analyzer = MedicalReportAnalyzer(model_name=model_name)
                            analysis = analyzer.analyze_report(medical_text)
                            display_analysis_results(analysis)
                        except Exception as e:
                            st.error(f"Error analyzing report: {str(e)}")
            else:
                st.warning("Currently only text files are supported. Please convert your file to text format.")
    
    elif input_method == "📋 Sample Reports":
        st.markdown('<h2 class="sub-header">📋 Sample Medical Reports</h2>', unsafe_allow_html=True)
        
        # Sample reports with kidney dialysis focus and varying health scores
        sample_reports = {
            "Kidney Dialysis Patient - Stable": """
PATIENT: Robert Wilson, Age: 58, Date: 2024-01-15
DIALYSIS STATUS: Hemodialysis 3x/week, 4 hours per session

COMPREHENSIVE METABOLIC PANEL:
Creatinine: 8.2 mg/dL (Normal: 0.7-1.3)
eGFR: 8 mL/min/1.73m² (Normal: >90)
BUN: 45 mg/dL (Normal: 7-20)
Sodium: 138 mEq/L (Normal: 135-145)
Potassium: 4.8 mEq/L (Normal: 3.5-5.0)
Calcium: 8.9 mg/dL (Normal: 8.5-10.5)
Phosphorus: 4.2 mg/dL (Normal: 2.5-4.5)
Albumin: 3.8 g/dL (Normal: 3.5-5.0)

COMPLETE BLOOD COUNT:
Hemoglobin: 11.2 g/dL (Normal: 13.5-17.5)
White Blood Cells: 6,800/μL (Normal: 4,500-11,000)
Platelets: 220,000/μL (Normal: 150,000-450,000)

DIALYSIS ACCESS:
- Left forearm AV fistula, patent and functioning
- Blood flow rate: 350 mL/min
- Dialysis adequacy (Kt/V): 1.4 (Target: >1.2)

MEDICATIONS: Erythropoietin 10,000 units weekly, Calcitriol 0.25mcg daily, PhosLo 667mg with meals
ALLERGIES: None known

NOTES: Patient is stable on hemodialysis with good compliance. Dialysis adequacy is within target range. Hemoglobin levels are maintained with erythropoietin support.
            """,
            
            "Kidney Dialysis Patient - Critical": """
PATIENT: Maria Rodriguez, Age: 62, Date: 2024-01-20
DIALYSIS STATUS: Hemodialysis 3x/week, recent missed sessions

COMPREHENSIVE METABOLIC PANEL:
Creatinine: 12.5 mg/dL (Normal: 0.7-1.3)
eGFR: 5 mL/min/1.73m² (Normal: >90)
BUN: 78 mg/dL (Normal: 7-20)
Sodium: 132 mEq/L (Normal: 135-145)
Potassium: 6.8 mEq/L (Normal: 3.5-5.0) - CRITICAL
Calcium: 7.2 mg/dL (Normal: 8.5-10.5)
Phosphorus: 8.5 mg/dL (Normal: 2.5-4.5)
Albumin: 2.8 g/dL (Normal: 3.5-5.0)

COMPLETE BLOOD COUNT:
Hemoglobin: 8.5 g/dL (Normal: 13.5-17.5) - SEVERE ANEMIA
White Blood Cells: 15,200/μL (Normal: 4,500-11,000)
Platelets: 95,000/μL (Normal: 150,000-450,000)

DIALYSIS ACCESS:
- Right upper arm AV graft, showing signs of stenosis
- Blood flow rate: 180 mL/min (reduced)
- Dialysis adequacy (Kt/V): 0.8 (Target: >1.2) - INADEQUATE

CARDIAC MARKERS:
Troponin I: 0.08 ng/mL (Normal: <0.04)
BNP: 1,200 pg/mL (Normal: <100)

MEDICATIONS: Erythropoietin 20,000 units weekly, Calcitriol 0.5mcg daily
ALLERGIES: Penicillin

NOTES: CRITICAL - Patient missed 2 dialysis sessions. Severe hyperkalemia requiring immediate dialysis. AV graft stenosis affecting dialysis adequacy. Severe anemia and fluid overload present.
            """,
            
            "Kidney Transplant Candidate": """
PATIENT: David Thompson, Age: 45, Date: 2024-01-18
DIALYSIS STATUS: Peritoneal dialysis, 4 exchanges daily

COMPREHENSIVE METABOLIC PANEL:
Creatinine: 6.8 mg/dL (Normal: 0.7-1.3)
eGFR: 12 mL/min/1.73m² (Normal: >90)
BUN: 32 mg/dL (Normal: 7-20)
Sodium: 140 mEq/L (Normal: 135-145)
Potassium: 4.1 mEq/L (Normal: 3.5-5.0)
Calcium: 9.2 mg/dL (Normal: 8.5-10.5)
Phosphorus: 3.8 mg/dL (Normal: 2.5-4.5)
Albumin: 4.1 g/dL (Normal: 3.5-5.0)

COMPLETE BLOOD COUNT:
Hemoglobin: 12.8 g/dL (Normal: 13.5-17.5)
White Blood Cells: 7,200/μL (Normal: 4,500-11,000)
Platelets: 280,000/μL (Normal: 150,000-450,000)

PERITONEAL DIALYSIS:
- Tenckhoff catheter in place, functioning well
- Peritoneal equilibration test: High transporter
- Ultrafiltration: 800-1000 mL/day
- Dialysis adequacy (Kt/V): 2.1 (Target: >1.7)

MEDICATIONS: Erythropoietin 8,000 units weekly, Calcitriol 0.25mcg daily
ALLERGIES: Sulfa drugs

NOTES: Excellent candidate for kidney transplantation. Well-controlled on peritoneal dialysis with good compliance and adequate clearance. Hemoglobin levels near normal range.
            """,
            
            "Kidney Disease - Pre-Dialysis": """
PATIENT: Jennifer Lee, Age: 51, Date: 2024-01-22
DIALYSIS STATUS: Not yet on dialysis, CKD Stage 4

COMPREHENSIVE METABOLIC PANEL:
Creatinine: 3.2 mg/dL (Normal: 0.7-1.3)
eGFR: 18 mL/min/1.73m² (Normal: >90)
BUN: 28 mg/dL (Normal: 7-20)
Sodium: 139 mEq/L (Normal: 135-145)
Potassium: 5.2 mEq/L (Normal: 3.5-5.0)
Calcium: 9.0 mg/dL (Normal: 8.5-10.5)
Phosphorus: 5.8 mg/dL (Normal: 2.5-4.5)
Albumin: 3.9 g/dL (Normal: 3.5-5.0)

COMPLETE BLOOD COUNT:
Hemoglobin: 10.8 g/dL (Normal: 13.5-17.5)
White Blood Cells: 8,500/μL (Normal: 4,500-11,000)
Platelets: 240,000/μL (Normal: 150,000-450,000)

URINALYSIS:
Protein: 2+ (Normal: Negative)
Blood: 1+ (Normal: Negative)
Specific Gravity: 1.015

MEDICATIONS: Lisinopril 10mg daily, Furosemide 40mg daily, Sodium bicarbonate 650mg twice daily
ALLERGIES: None known

NOTES: CKD Stage 4 with declining kidney function. Patient education about dialysis options initiated. Blood pressure and diabetes well-controlled. Preparing for eventual dialysis initiation.
            """,
            
            "Cardiac Assessment": """
PATIENT: Sarah Johnson, Age: 52, Date: 2024-01-20

CARDIAC MARKERS:
Troponin I: 0.15 ng/mL (Normal: <0.04)
BNP: 450 pg/mL (Normal: <100)
CK-MB: 25 ng/mL (Normal: <5.0)

ELECTROCARDIOGRAM:
- Sinus rhythm with occasional PVCs
- ST-segment depression in leads II, III, aVF
- T-wave inversions in anterior leads

ECHOCARDIOGRAM:
- Left ventricular ejection fraction: 45% (Normal: 55-70%)
- Regional wall motion abnormalities
- Mild mitral regurgitation

DIAGNOSIS: Acute coronary syndrome with reduced ejection fraction
RECOMMENDATIONS: Immediate cardiology consultation, cardiac catheterization
            """
        }
        
        selected_sample = st.selectbox(
            "Choose a sample report:",
            list(sample_reports.keys())
        )
        
        if selected_sample:
            st.text_area(
                "Sample Report Content:",
                sample_reports[selected_sample],
                height=300,
                disabled=True
            )
            
            if st.button("🔍 Analyze Sample Report", type="primary"):
                with st.spinner("Analyzing sample report..."):
                    try:
                        analyzer = MedicalReportAnalyzer(model_name=model_name)
                        analysis = analyzer.analyze_report(sample_reports[selected_sample])
                        display_analysis_results(analysis)
                    except Exception as e:
                        st.error(f"Error analyzing report: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>⚠️ <strong>Disclaimer:</strong> This AI tool is for educational and informational purposes only. 
        It should not replace professional medical advice, diagnosis, or treatment. 
        Always consult with qualified healthcare providers for medical decisions.</p>
        <p style="margin-top: 1rem; font-size: 0.9rem;">© 2024 DCC Medical Report Agent - Powered by Advanced AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 