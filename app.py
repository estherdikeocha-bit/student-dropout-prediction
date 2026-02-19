"""
STUDENT DROPOUT PREDICTION - STREAMLIT APP
===========================================
Interactive web app for predicting student dropout risk
"""

import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

# CSS styling with Light Green Background
st.markdown("""
    <style>
    /* Main background - Light Green */
    .stApp {
        background-color: #E8F5E9;
    }
    
    /* Sidebar background - Slightly Darker Green */
    [data-testid="stSidebar"] {
        background-color: #C8E6C9;
    }
    
    .main-header {
        font-size: 3rem;
        color: #1B5E20;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        color: #2E7D32;
        text-align: center;
        font-size: 1.2rem;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #D32F2F 0%, #F44336 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(211, 47, 47, 0.3);
    }
    
    .moderate-risk {
        background: linear-gradient(135deg, #F57C00 0%, #FF9800 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(245, 124, 0, 0.3);
    }
    
    .low-risk {
        background: linear-gradient(135deg, #388E3C 0%, #4CAF50 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(56, 142, 60, 0.3);
    }
    
    .risk-percentage {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* Metric cards on landing page */
    [data-testid="stMetricValue"] {
        color: #1B5E20;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎓 Student Dropout Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Early Warning System for University Student Retention</p>', unsafe_allow_html=True)

# Load model silently
@st.cache_resource
def load_trained_model():
    return load_model('student_dropout_model')

try:
    model = load_trained_model()
except:
    st.error(" Model not found! Please run 'python train.py' first.")
    st.stop()

# Sidebar - Student Information
st.sidebar.header("👤 Student Information")

# Demographics
st.sidebar.subheader("📋 Demographics")
age = st.sidebar.slider("Age", 16, 35, 20)
gender = st.sidebar.radio("Gender", ['Male', 'Female'])
state_of_origin = st.sidebar.selectbox(
    "State of Origin",
    ['Lagos', 'Kano', 'Rivers', 'Oyo', 'Anambra', 'Kaduna', 
     'Enugu', 'Ogun', 'Delta', 'Edo', 'Katsina', 'Borno', 
     'Cross_River', 'Imo', 'Ekiti']
)
distance_from_home_km = st.sidebar.slider("Distance from Home (km)", 10, 800, 200)
marital_status = st.sidebar.radio("Marital Status", ['Single', 'Married'])
has_children = st.sidebar.checkbox("Has Children")

# Academic Background
st.sidebar.subheader("📚 Academic Background")
admission_score = st.sidebar.slider("Admission Score (JAMB)", 180, 350, 250)
secondary_school_type = st.sidebar.radio("Secondary School Type", ['Public', 'Private'])
secondary_cgpa = st.sidebar.slider("Secondary School CGPA", 2.0, 5.0, 3.5, 0.1)

# Current Academic Status
st.sidebar.subheader("🎯 Current Academic Status")
year_of_study = st.sidebar.selectbox("Year of Study", [1, 2, 3, 4, 5])
current_cgpa = st.sidebar.slider("Current CGPA", 1.5, 5.0, 3.0, 0.1, help="Most important predictor!")

# CGPA status indicator
if current_cgpa >= 4.5:
    st.sidebar.success("🌟 First Class!")
elif current_cgpa >= 3.5:
    st.sidebar.success("✨ Second Class Upper")
elif current_cgpa >= 2.5:
    st.sidebar.info("📘 Second Class Lower")
elif current_cgpa >= 2.0:
    st.sidebar.warning("⚠️ Third Class")
else:
    st.sidebar.error("🚨 Below Pass Grade")

course_load_per_semester = st.sidebar.slider("Course Load (Credits)", 12, 30, 18)
attendance_percentage = st.sidebar.slider("Attendance Percentage", 30, 100, 75)
number_of_failed_courses = st.sidebar.number_input("Failed Courses", 0, 20, 2)
number_of_repeated_courses = st.sidebar.number_input("Repeated Courses", 0, 10, 0)
semester_gpa_trend = st.sidebar.select_slider(
    "Semester GPA Trend",
    options=['Declining', 'Stable', 'Improving'],
    value='Stable'
)
previous_warnings = st.sidebar.number_input("Academic Warnings", 0, 5, 0)
probation_status = st.sidebar.checkbox("Currently on Academic Probation")

# Program Details
st.sidebar.subheader("🏫 Program Details")
department = st.sidebar.selectbox(
    "Department",
    ['Engineering', 'Medicine', 'Law', 'Science', 'Arts',
     'Social_Sciences', 'Education', 'Business_Admin',
     'Agriculture', 'Environmental_Studies']
)
program_difficulty = st.sidebar.select_slider(
    "Program Difficulty",
    options=['Low', 'Moderate', 'High', 'Very_High'],
    value='Moderate'
)
is_stem = 1 if department in ['Engineering', 'Medicine', 'Science'] else 0

# Financial Situation
st.sidebar.subheader("💰 Financial Situation")
scholarship_status = st.sidebar.selectbox("Scholarship Status", ['None', 'Partial', 'Full'])
family_income_level = st.sidebar.selectbox("Family Income Level", ['Low', 'Medium', 'High'])
fee_payment_status = st.sidebar.selectbox("Fee Payment Status", ['Fully_Paid', 'Partially_Paid', 'Owing'])
has_part_time_job = st.sidebar.checkbox("Has Part-time Job")
receives_allowance = st.sidebar.checkbox("Receives Regular Allowance")
financial_stress_level = st.sidebar.select_slider(
    "Financial Stress Level",
    options=['Low', 'Moderate', 'High', 'Very_High'],
    value='Moderate'
)

# Academic Engagement
st.sidebar.subheader("📖 Academic Engagement")
library_visits_per_week = st.sidebar.slider("Library Visits per Week", 0, 15, 3)
online_platform_usage_hours = st.sidebar.slider("LMS/E-Learning Hours per Week", 0, 30, 5)
participation_in_clubs = st.sidebar.checkbox("Participates in Student Clubs")
has_mentor = st.sidebar.checkbox("Has Academic/Personal Mentor")
peer_study_groups = st.sidebar.checkbox("Joins Peer Study Groups")
social_integration_score = st.sidebar.slider("Social Integration Score", 1, 10, 6, help="1=Isolated, 10=Well integrated")

# Support & Wellbeing
st.sidebar.subheader("🏥 Support & Wellbeing")
received_academic_counseling = st.sidebar.checkbox("Received Academic Counseling")
tutoring_sessions_attended = st.sidebar.slider("Tutoring Sessions Attended", 0, 20, 2)
accommodation_type = st.sidebar.selectbox(
    "Accommodation Type",
    ['Campus_Hostel', 'Off_Campus', 'Living_with_Family', 'Private_Hostel']
)
health_status = st.sidebar.selectbox("Health Status", ['Excellent', 'Good', 'Fair', 'Poor'])
stress_level = st.sidebar.select_slider(
    "Stress Level",
    options=['Low', 'Moderate', 'High', 'Very_High'],
    value='Moderate'
)

# Personal Factors
st.sidebar.subheader(" Personal Factors")
motivation_level = st.sidebar.select_slider(
    "Motivation Level",
    options=['Very_Low', 'Low', 'Moderate', 'High', 'Very_High'],
    value='Moderate'
)
career_clarity = st.sidebar.select_slider(
    "Career Goal Clarity",
    options=['Very_Unclear', 'Unclear', 'Somewhat_Clear', 'Clear', 'Very_Clear'],
    value='Somewhat_Clear'
)
family_support = st.sidebar.select_slider(
    "Family Support Level",
    options=['Very_Low', 'Low', 'Moderate', 'High', 'Very_High'],
    value='Moderate'
)

# Predict Button
if st.sidebar.button("🔍 Assess Dropout Risk", type="primary"):
    
    # Create input DataFrame with ALL 41 features
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'state_of_origin': [state_of_origin],
        'distance_from_home_km': [distance_from_home_km],
        'marital_status': [marital_status],
        'has_children': [1 if has_children else 0],
        'admission_score': [admission_score],
        'secondary_school_type': [secondary_school_type],
        'secondary_cgpa': [secondary_cgpa],
        'year_of_study': [year_of_study],
        'current_cgpa': [current_cgpa],
        'course_load_per_semester': [course_load_per_semester],
        'attendance_percentage': [attendance_percentage],
        'number_of_failed_courses': [number_of_failed_courses],
        'number_of_repeated_courses': [number_of_repeated_courses],
        'semester_gpa_trend': [semester_gpa_trend],
        'department': [department],
        'program_difficulty': [program_difficulty],
        'scholarship_status': [scholarship_status],
        'family_income_level': [family_income_level],
        'fee_payment_status': [fee_payment_status],
        'has_part_time_job': [1 if has_part_time_job else 0],
        'receives_allowance': [1 if receives_allowance else 0],
        'financial_stress_level': [financial_stress_level],
        'library_visits_per_week': [library_visits_per_week],
        'online_platform_usage_hours': [online_platform_usage_hours],
        'participation_in_clubs': [1 if participation_in_clubs else 0],
        'has_mentor': [1 if has_mentor else 0],
        'peer_study_groups': [1 if peer_study_groups else 0],
        'social_integration_score': [social_integration_score],
        'accommodation_type': [accommodation_type],
        'health_status': [health_status],
        'stress_level': [stress_level],
        'received_academic_counseling': [1 if received_academic_counseling else 0],
        'tutoring_sessions_attended': [tutoring_sessions_attended],
        'motivation_level': [motivation_level],
        'career_clarity': [career_clarity],
        'family_support': [family_support],
        'previous_warnings': [previous_warnings],
        'probation_status': [1 if probation_status else 0],
        'is_stem': [is_stem]
    })
    
    # Make prediction
    prediction = predict_model(model, data=input_data)
    dropout_probability = prediction['prediction_score'].values[0]
    
    # Determine risk level
    if dropout_probability >= 0.7:
        risk_level = "HIGH RISK"
        card_class = "high-risk"
        risk_color = "#F44336"
    elif dropout_probability >= 0.4:
        risk_level = "MODERATE RISK"
        card_class = "moderate-risk"
        risk_color = "#FF9800"
    else:
        risk_level = "LOW RISK"
        card_class = "low-risk"
        risk_color = "#4CAF50"
    
    # Display Results
    st.markdown("---")
    
    # Risk Card
    st.markdown(f"""
        <div class="{card_class}">
            <h1 style="margin: 0;"> {risk_level}</h1>
            <div class="risk-percentage">{dropout_probability*100:.1f}%</div>
            <p style="font-size: 1.2rem; margin: 0;">Probability of Dropping Out</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Risk Factors Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Risk Factors")
        risk_factors = []
        
        if current_cgpa < 2.5:
            risk_factors.append(f"⚠️ **Low CGPA**: {current_cgpa:.2f} (Below 2.5)")
        if attendance_percentage < 70:
            risk_factors.append(f"⚠️ **Poor Attendance**: {attendance_percentage}%")
        if number_of_failed_courses >= 5:
            risk_factors.append(f"⚠️ **Many Failures**: {number_of_failed_courses} courses")
        if fee_payment_status == 'Owing':
            risk_factors.append("⚠️ **Owing Fees**: Financial barrier")
        if probation_status:
            risk_factors.append("⚠️ **On Probation**: Critical academic status")
        if semester_gpa_trend == 'Declining':
            risk_factors.append("⚠️ **GPA Declining**: Negative trend")
        if motivation_level in ['Very_Low', 'Low']:
            risk_factors.append(f"⚠️ **Low Motivation**: {motivation_level}")
        if social_integration_score <= 3:
            risk_factors.append(f"⚠️ **Socially Isolated**: Score {social_integration_score}/10")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success(" No major risk factors identified!")
    
    with col2:
        st.markdown("### 🛡️ Protective Factors")
        protective = []
        
        if current_cgpa >= 3.5:
            protective.append(f" **Strong CGPA**: {current_cgpa:.2f}")
        if has_mentor:
            protective.append(" **Has Mentor**: Support system in place")
        if scholarship_status == 'Full':
            protective.append(" **Full Scholarship**: Financial security")
        if attendance_percentage >= 85:
            protective.append(f" **Excellent Attendance**: {attendance_percentage}%")
        if peer_study_groups:
            protective.append(" **Study Groups**: Peer support")
        if family_support in ['High', 'Very_High']:
            protective.append(f" **Strong Family Support**: {family_support}")
        if social_integration_score >= 7:
            protective.append(f" **Well Integrated**: Score {social_integration_score}/10")
        
        if protective:
            for factor in protective:
                st.success(factor)
        else:
            st.warning("⚠️ Limited protective factors - needs more support")
    
    # Recommended Interventions
    st.markdown("### 💡 Recommended Interventions")
    
    if risk_level == "HIGH RISK":
        st.error("🚨 URGENT ACTION REQUIRED")
        st.markdown("""
        **Immediate Interventions:**
        1. 📚 **Intensive Tutoring**: 10-15 hours per week
        2. 💰 **Emergency Financial Aid**: Assess fee payment options
        3. 👨‍🏫 **Weekly Advisor Meetings**: Monitor progress closely
        4. 🤝 **Assign Faculty Mentor**: Immediate mentorship
        5. 🧠 **Counseling Referral**: Address stress and motivation
        6. 👥 **Peer Support Groups**: Connect with study groups
        
        **Follow-up:** Weekly check-ins for 30 days, then bi-weekly
        """)
        
    elif risk_level == "MODERATE RISK":
        st.warning("⚠️ PROACTIVE SUPPORT NEEDED")
        st.markdown("""
        **Recommended Actions:**
        1. 📚 **Group Tutoring**: 5 hours per week
        2. 💰 **Financial Counseling**: Budget planning and scholarship search
        3. 📊 **Bi-weekly Progress Meetings**: Track improvements
        4. 👥 **Social Integration**: Encourage club participation
        5. 📖 **Study Skills Workshop**: Time management and learning strategies
        
        **Follow-up:** Bi-weekly check-ins, monthly progress reviews
        """)
        
    else:
        st.success(" STUDENT ON TRACK")
        st.markdown("""
        **Growth Opportunities:**
        1. 🎓 **Research Opportunities**: Undergraduate research programs
        2. 🌟 **Leadership Development**: Student government, peer mentoring
        3. 💼 **Career Preparation**: Internships, networking events
        4. 🏆 **Scholarship Nominations**: Awards and recognition
        5. 📚 **Advanced Courses**: Honors programs, graduate prep
        
        **Monitoring:** Semester check-ins to maintain positive trajectory
        """)
    
    # Risk Profile Visualization
    st.markdown("### 📊 Risk Profile Breakdown")
    
    # Calculate risk scores by category
    categories = ['Academic', 'Financial', 'Engagement', 'Personal', 'Support']
    
    # Academic risk
    academic_risk = 0
    if current_cgpa < 2.5: academic_risk += 40
    if number_of_failed_courses > 4: academic_risk += 30
    if attendance_percentage < 70: academic_risk += 20
    if semester_gpa_trend == 'Declining': academic_risk += 10
    
    # Financial risk
    financial_risk = 0
    if fee_payment_status == 'Owing': financial_risk += 40
    if financial_stress_level in ['High', 'Very_High']: financial_risk += 30
    if scholarship_status == 'None' and family_income_level == 'Low': financial_risk += 30
    
    # Engagement risk
    engagement_risk = 0
    if library_visits_per_week < 2: engagement_risk += 25
    if online_platform_usage_hours < 5: engagement_risk += 25
    if social_integration_score < 5: engagement_risk += 30
    if not participation_in_clubs: engagement_risk += 20
    
    # Personal risk
    personal_risk = 0
    if motivation_level in ['Very_Low', 'Low']: personal_risk += 40
    if stress_level in ['High', 'Very_High']: personal_risk += 30
    if family_support in ['Very_Low', 'Low']: personal_risk += 20
    if career_clarity in ['Very_Unclear', 'Unclear']: personal_risk += 10
    
    # Support deficit
    support_deficit = 100
    if has_mentor: support_deficit -= 30
    if received_academic_counseling: support_deficit -= 20
    if peer_study_groups: support_deficit -= 20
    if tutoring_sessions_attended > 5: support_deficit -= 30
    
    scores = [
        min(100, academic_risk),
        min(100, financial_risk),
        min(100, engagement_risk),
        min(100, personal_risk),
        max(0, support_deficit)
    ]
    
    # Create radar chart
    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor=risk_color,
        opacity=0.6,
        line=dict(color=risk_color, width=2),
        name='Risk Level'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="Risk Profile (0=Low Risk, 100=High Risk)",
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Contact Information
    st.markdown("### 📞 Get Help Now")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **University Support Services:**
        - 📧 Academic Advising: advising@university.edu
        - 💰 Financial Aid: finaid@university.edu
        - 🧠 Counseling Center: counseling@university.edu
        - 📚 Learning Center: tutoring@university.edu
        """)
    
    with col2:
        st.markdown("""
        **Emergency Contacts:**
        - 🆘 24/7 Crisis Line: 0800-HELP-NOW
        - 👨‍⚕️ Health Services: health@university.edu
        - 🏠 Housing Office: housing@university.edu
        - 👮 Campus Security: (123) 456-7890
        """)

else:
    # Landing Page
    st.info("👈 **Enter student information** in the sidebar and click **'Assess Dropout Risk'** to generate prediction")
    
    st.markdown("###  Why Use This System?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nigerian Dropout Rate", "22%", help="Percentage of students who drop out")
    
    with col2:
        st.metric("Year 1 Critical", "40%", help="Percentage of dropouts occur in first year")
    
    with col3:
        st.metric("Top Cause", "Financial", help="Leading cause of dropout")
    
    st.markdown("### 💡 Impact of Early Intervention")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Student Success Rates with Intervention:**
        -  **75% retention rate** (vs 20% without intervention)
        - 📈 **1.5 point CGPA improvement** on average
        - 📊 **60% reduction** in course failures
        - 🎓 **3x more likely** to graduate on time
        """)
    
    with col2:
        st.markdown("""
        **Business Impact for Universities:**
        - 💰 **₦1.3B saved annually** per 1,000 students
        - 📊 **15-25% improvement** in graduation rates
        - ⭐ **Higher rankings** and reputation
        - 🎯 **Efficient resource allocation**
        """)
    
    st.success("🚀 **This system helps save students before it's too late!**")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #1B5E20;">
        <p style="font-size: 1.1rem; font-weight: bold;">🎓 Student Dropout Prediction System</p>
        <p>Built with ❤️ by Blossom Academy | Powered by PyCaret & AI</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            💡 Helping Nigerian universities improve student retention and graduation rates
        </p>
    </div>
""", unsafe_allow_html=True)