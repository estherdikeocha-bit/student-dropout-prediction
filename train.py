"""
STUDENT DROPOUT PREDICTION - PYCARET TRAINING
==============================================
Clean, simple training script with PyCaret
"""

from pycaret.classification import *
import pandas as pd

print("="*70)
print("STUDENT DROPOUT PREDICTION - TRAINING WITH PYCARET")
print("="*70)

# Load data
print("\n[1/6] Loading dataset...")
data = pd.read_csv("student_dropout_dataset.csv")
print(f" Loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# Fill missing values
print("\n[2/6] Cleaning data...")
data["scholarship_status"] = data["scholarship_status"].fillna("None")
print(f" Filled missing scholarship_status values")

# Initialize PyCaret
print("\n[3/6] Setting up PyCaret...")
clf = setup(
    data=data,
    target='will_dropout',
    session_id=42,
    fix_imbalance=True,              # SMOTE for class balance
    normalize=True,                   # Normalize features
    fold=5,                          # 5-fold CV
    categorical_features=[
        'gender', 'state_of_origin', 'marital_status', 
        'secondary_school_type', 'semester_gpa_trend', 
        'department', 'program_difficulty', 'scholarship_status',
        'family_income_level', 'fee_payment_status', 
        'financial_stress_level', 'accommodation_type',
        'health_status', 'stress_level', 'motivation_level',
        'career_clarity', 'family_support'
    ],
    numeric_features=[
        'age', 'distance_from_home_km', 'has_children',
        'admission_score', 'secondary_cgpa', 'year_of_study',
        'current_cgpa', 'course_load_per_semester',
        'attendance_percentage', 'number_of_failed_courses',
        'number_of_repeated_courses', 'has_part_time_job',
        'receives_allowance', 'library_visits_per_week',
        'online_platform_usage_hours', 'participation_in_clubs',
        'has_mentor', 'peer_study_groups', 'social_integration_score',
        'received_academic_counseling', 'tutoring_sessions_attended',
        'previous_warnings', 'probation_status', 'is_stem'
    ]
)
print(" Setup complete!")

# Compare models
print("\n[4/6] Comparing models...")
best_model = compare_models(sort='Recall', n_select=1)
print(f" Best model selected: {type(best_model).__name__}")

# Tune model
print("\n[5/6] Tuning hyperparameters...")
tuned_model = tune_model(best_model, optimize='Recall', n_iter=20)
tuned_results = pull()
print(f" Tuned model - Recall: {tuned_results.loc['Mean', 'Recall']:.4f}")

# Finalize model
print("\n[6/6] Finalizing and saving...")
final_model = finalize_model(tuned_model)
save_model(final_model, "student_dropout_model")

print("\n" + "="*70)
print(" MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n Model saved as: student_dropout_model.pkl")
print(f" Final Recall: {tuned_results.loc['Mean', 'Recall']*100:.2f}%")
print(f" Final F1: {tuned_results.loc['Mean', 'F1']*100:.2f}%")
print(f"\n Ready for deployment!")
print("   Run: streamlit run app.py")
print("="*70)