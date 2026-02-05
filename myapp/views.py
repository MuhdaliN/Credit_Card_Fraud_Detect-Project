# views.py 
import base64
import csv
import json
import os
import uuid
from datetime import datetime
from io import BytesIO

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.hashers import check_password, make_password
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST
from sklearn.linear_model import LinearRegression

from .ml_utils import (extract_ml_features, preprocess_bank_statement,
                       predict_fraud_ml)
from .models import UserCSVFile, User_db

ML_MODELS_LOADED = False
model = None
scaler = None
feature_names = None


def load_ml_models():
    """Load ML models once when Django starts"""
    global ML_MODELS_LOADED, model, scaler, feature_names
    
    try:
        model_dir = os.path.join(settings.BASE_DIR, "myproject", "ml")
        model_path = os.path.join(model_dir, "model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            ML_MODELS_LOADED = True
            print("✓ ML models loaded successfully")
            
            feature_names_path = os.path.join(model_dir, "feature_names.pkl")
            if os.path.exists(feature_names_path):
                feature_names = joblib.load(feature_names_path)
                print(f"✓ Feature names loaded: {len(feature_names)} features")
        else:
            print("✗ ML model files not found")
            ML_MODELS_LOADED = False
            
    except Exception as e:
        print(f"✗ Error loading ML models: {str(e)}")
        ML_MODELS_LOADED = False


load_ml_models()


def home(request):
    return render(request, "home.html")


def register_view(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm = request.POST.get('confirm')

        if not all([name, email, password, confirm]):
            messages.error(request, "All fields are required")
            return redirect('register')
        
        if password != confirm:
            messages.error(request, "Passwords do not match")
            return redirect('register')
        
        if User_db.objects.filter(email_db=email).exists():
            messages.error(request, "Email already registered")
            return redirect('register')

        student = User_db(
            name_db=name,
            email_db=email,
            pass_db=make_password(password)
        )
        student.save()
        messages.success(request, "Registration successful! Please login.")
        return redirect('login')

    return render(request, 'register.html')


def login_view(request):
    errors = {}
    if request.method == "POST":
        email = request.POST.get('email_login')
        password = request.POST.get('pass_login')
        remember_me = request.POST.get('remember') == 'on'

        if not email or not password:
            messages.error(request, "Email and password are required")
            return render(request, "login.html", {'errors': errors})

        try:
            user = User_db.objects.get(email_db=email)

            if check_password(password, user.pass_db):
                request.session['logged_user'] = user.email_db
                request.session['name'] = user.name_db
                request.session['user_id'] = user.id

                response = redirect('dashboard')

                if remember_me:
                    response.set_cookie('logged_user', user.email_db, max_age=7*24*60*60)
                else:
                    response.delete_cookie('logged_user')

                return response
            else:
                errors['pass_error'] = "Wrong password!"
        except User_db.DoesNotExist:
            errors['email_error'] = "User does not exist!"

    return render(request, "login.html", {'errors': errors})


def dashboard_view(request):
    if 'logged_user' not in request.session:
        if 'logged_user' in request.COOKIES:
            try:
                user = User_db.objects.get(email_db=request.COOKIES['logged_user'])
                request.session['logged_user'] = user.email_db
                request.session['name'] = user.name_db
                request.session['user_id'] = user.id
            except User_db.DoesNotExist:
                return redirect('login')
        else:
            return redirect('login')
    
    user_email = request.session['logged_user']
    user = User_db.objects.get(email_db=user_email)
    csv_file = None
    
    try:
        csv_file = UserCSVFile.objects.get(user=user)
    except UserCSVFile.DoesNotExist:
        csv_file = None
    
    global ML_MODELS_LOADED
    if not ML_MODELS_LOADED:
        print("Attempting to reload ML models...")
        load_ml_models()
    
    context = {
        'user': user,
        'csv_file': csv_file,
        'ml_models_loaded': ML_MODELS_LOADED,
    }
    
    return render(request, 'dashboard.html', context)


def upload_csv_view(request):
    if 'logged_user' not in request.session:
        return redirect('login')
    
    if request.method == 'POST' and request.FILES.get('csv_file'):
        user_email = request.session['logged_user']
        user = User_db.objects.get(email_db=user_email)
        uploaded_file = request.FILES['csv_file']
        
        if not uploaded_file.name.lower().endswith('.csv'):
            messages.error(request, "Please upload a CSV file only.")
            return redirect('dashboard')
        
        if uploaded_file.size > 10 * 1024 * 1024:
            messages.error(request, "File size should be less than 10MB.")
            return redirect('dashboard')
        
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                messages.error(request, "CSV file is empty.")
                return redirect('dashboard')
            
            original_filename = uploaded_file.name
            unique_filename = f"{user.id}_{uuid.uuid4().hex[:8]}_{original_filename}"
            
            try:
                csv_record = UserCSVFile.objects.get(user=user)
                
                if csv_record.file and os.path.exists(csv_record.file.path):
                    os.remove(csv_record.file.path)
                
                csv_record.file.delete(save=False)
                csv_record.file.save(unique_filename, uploaded_file, save=False)
                csv_record.original_filename = original_filename
                csv_record.save()
                messages.success(request, f"CSV file '{original_filename}' updated successfully!")
                
            except UserCSVFile.DoesNotExist:
                csv_record = UserCSVFile(
                    user=user,
                    original_filename=original_filename
                )
                csv_record.file.save(unique_filename, uploaded_file, save=False)
                csv_record.save()
                messages.success(request, f"CSV file '{original_filename}' uploaded successfully!")
            
            return redirect('dashboard')
            
        except pd.errors.EmptyDataError:
            messages.error(request, "CSV file is empty.")
            return redirect('dashboard')
        except pd.errors.ParserError:
            messages.error(request, "Invalid CSV file format.")
            return redirect('dashboard')
        except Exception as e:
            messages.error(request, f"Error processing CSV file: {str(e)}")
            return redirect('dashboard')
    
    return redirect('dashboard')


def delete_csv_view(request):
    if 'logged_user' not in request.session:
        return redirect('login')
    
    user_email = request.session['logged_user']
    user = User_db.objects.get(email_db=user_email)
    
    try:
        csv_record = UserCSVFile.objects.get(user=user)
        
        if csv_record.file and os.path.exists(csv_record.file.path):
            os.remove(csv_record.file.path)
        
        csv_record.delete()
        messages.success(request, "CSV file deleted successfully!")
        
    except UserCSVFile.DoesNotExist:
        messages.error(request, "No CSV file found to delete.")
    
    return redirect('dashboard')


def faq_view(request):
    return render(request, "faq.html")


def logout_view(request):
    logout(request)
    return redirect('home')


def advanced_analysis_view(request):
    """Perform advanced fraud analysis on uploaded CSV"""
    if 'logged_user' not in request.session:
        return redirect('login')
    
    user_email = request.session['logged_user']
    user = User_db.objects.get(email_db=user_email)
    
    try:
        csv_record = UserCSVFile.objects.get(user=user)
        
        # Read and preprocess CSV
        df = pd.read_csv(csv_record.file.path)
        df_processed = preprocess_bank_statement(df)
        
        # Ensure required columns exist
        if 'Heuristic_Risk_Score' not in df_processed.columns:
            df_processed['Heuristic_Risk_Score'] = 0
        if 'Heuristic_Fraud_Flag' not in df_processed.columns:
            df_processed['Heuristic_Fraud_Flag'] = 'LOW RISK'
        if 'Transaction_Type' not in df_processed.columns:
            df_processed['Transaction_Type'] = 'DEBIT'
        
        # Extract features
        features = extract_ml_features(df_processed)
        if not features:  # Check if features is empty or None
            features = {}
        
        # Get ML prediction
        try:
            fraud_probability = predict_fraud_ml(features)
        except Exception as e:
            print(f"ML prediction error: {e}")
            fraud_probability = 10.0  # Default fallback
        
        # Generate visualizations
        charts = generate_analysis_charts(df_processed)
        
        # Get top risky transactions
        risky_transactions = []
        if not df_processed.empty and 'Heuristic_Risk_Score' in df_processed.columns:
            risky_df = df_processed.nlargest(10, 'Heuristic_Risk_Score')
            
            # Ensure all required columns exist
            for col in ['Date', 'Details', 'Amount', 'Heuristic_Risk_Score', 'Heuristic_Fraud_Flag']:
                if col not in risky_df.columns:
                    risky_df[col] = '' if col == 'Details' else 0
            
            # Convert to records
            risky_transactions = risky_df[
                ['Date', 'Details', 'Amount', 'Heuristic_Risk_Score', 'Heuristic_Fraud_Flag']
            ].to_dict('records')
            
            # Format dates
            for transaction in risky_transactions:
                if transaction['Date'] and pd.notna(transaction['Date']):
                    try:
                        transaction['Date'] = pd.to_datetime(transaction['Date']).date()
                    except:
                        transaction['Date'] = None
        
        # Calculate statistics
        total_transactions = len(df_processed)
        
        if 'Heuristic_Risk_Score' in df_processed.columns:
            high_risk_count = (df_processed['Heuristic_Risk_Score'] >= 70).sum()
            medium_risk_count = ((df_processed['Heuristic_Risk_Score'] >= 40) & 
                               (df_processed['Heuristic_Risk_Score'] < 70)).sum()
            avg_risk_score = float(df_processed['Heuristic_Risk_Score'].mean())
        else:
            high_risk_count = 0
            medium_risk_count = 0
            avg_risk_score = 0
        
        total_amount_at_risk = 0
        if 'Amount' in df_processed.columns and 'Heuristic_Risk_Score' in df_processed.columns:
            total_amount_at_risk = float(df_processed[df_processed['Heuristic_Risk_Score'] >= 40]['Amount'].sum())
        
        # Prepare analysis results with safe defaults
        analysis_result = {
            'fraud_probability': float(fraud_probability),
            'total_transactions': int(total_transactions),
            'high_risk_count': int(high_risk_count),
            'medium_risk_count': int(medium_risk_count),
            'total_amount_at_risk': float(total_amount_at_risk),
            'risk_score': float(avg_risk_score),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'charts': charts,
            'features': {
                'pct_high_risk': float(features.get('pct_high_risk', 0)),
                'pct_rapid_txns': float(features.get('pct_rapid_txns', 0)),
                'pct_unusual_hours': float(features.get('pct_unusual_hours', 0)),
                'pct_large_txns': float(features.get('pct_large_txns', 0)),
            }
        }
        
        # Save to database
        csv_record.analysis_result = analysis_result
        csv_record.save()
        
        context = {
            'analysis': analysis_result,
            'risky_transactions': risky_transactions,
            'charts': charts,
            'csv_file': csv_record,
        }
        
        return render(request, 'advanced_analysis.html', context)
        
    except UserCSVFile.DoesNotExist:
        messages.error(request, "Please upload a CSV file first")
        return redirect('dashboard')
    except Exception as e:
        print(f"Advanced analysis error: {e}")
        import traceback
        traceback.print_exc()
        messages.error(request, f"Analysis error: {str(e)}")
        return redirect('dashboard')


def generate_analysis_charts(df_processed):
    """Generate visualization charts for analysis"""
    charts = {}
    
    try:
        # Debug info
        print(f"In generate_analysis_charts - DataFrame columns: {list(df_processed.columns)}")
        print(f"Heuristic_Fraud_Flag in columns: {'Heuristic_Fraud_Flag' in df_processed.columns}")
        
        # 1. Risk Distribution Chart
        plt.figure(figsize=(8, 5))
        
        # Check if column exists
        if 'Heuristic_Fraud_Flag' in df_processed.columns:
            risk_counts = df_processed['Heuristic_Fraud_Flag'].value_counts()
            print(f"Risk counts: {risk_counts.to_dict()}")
        else:
            # Create default values
            risk_counts = pd.Series({
                'LOW RISK': len(df_processed),
                'MEDIUM RISK': 0,
                'HIGH RISK': 0
            })
            print("Using default risk counts")
        
        colors = ['#34d399', '#fbbf24', '#f87171']  # Green, Yellow, Red
        
        # Make sure we have all risk categories
        all_categories = ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']
        for cat in all_categories:
            if cat not in risk_counts.index:
                risk_counts[cat] = 0
        
        risk_counts = risk_counts.reindex(all_categories)
        
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Transaction Risk Distribution')
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        charts['risk_distribution'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Transaction Amount Over Time
        plt.figure(figsize=(10, 5))
        if 'Date' in df_processed.columns and 'Amount' in df_processed.columns:
            df_sorted = df_processed.sort_values('Date')
            plt.plot(df_sorted['Date'], df_sorted['Amount'], alpha=0.7)
            
            # Highlight high-risk transactions if column exists
            if 'Heuristic_Risk_Score' in df_sorted.columns:
                high_risk = df_sorted[df_sorted['Heuristic_Risk_Score'] >= 70]
                if not high_risk.empty:
                    plt.scatter(high_risk['Date'], high_risk['Amount'], 
                               color='red', s=50, label='High Risk', alpha=0.7)
        
        plt.title('Transaction Amount Over Time')
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        charts['amount_timeline'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 3. Hourly Transaction Pattern
        plt.figure(figsize=(8, 5))
        if 'Hour' in df_processed.columns:
            hourly_counts = df_processed['Hour'].value_counts().sort_index()
            plt.bar(hourly_counts.index, hourly_counts.values, alpha=0.7)
            plt.title('Transactions by Hour of Day')
            plt.xlabel('Hour (24-hour format)')
            plt.ylabel('Number of Transactions')
            # Set appropriate x-ticks
            if not hourly_counts.empty:
                plt.xticks(range(int(hourly_counts.index.min()), int(hourly_counts.index.max()) + 1))
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        charts['hourly_pattern'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 4. Risk Score Distribution
        plt.figure(figsize=(8, 5))
        if 'Heuristic_Risk_Score' in df_processed.columns:
            plt.hist(df_processed['Heuristic_Risk_Score'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            plt.title('Risk Score Distribution')
            plt.xlabel('Risk Score (0-100)')
            plt.ylabel('Frequency')
            plt.axvline(x=70, color='red', linestyle='--', label='High Risk Threshold')
            plt.axvline(x=40, color='yellow', linestyle='--', label='Medium Risk Threshold')
            plt.legend()
        else:
            # Create empty histogram
            plt.text(0.5, 0.5, 'No Risk Score Data', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.title('Risk Score Distribution - No Data')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        charts['risk_histogram'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
    except Exception as e:
        print(f"Chart generation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide placeholder images
        placeholder = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        charts = {
            'risk_distribution': placeholder,
            'amount_timeline': placeholder,
            'hourly_pattern': placeholder,
            'risk_histogram': placeholder
        }
    
    return charts


def download_analysis_report(request):
    """Download analysis report as CSV"""
    if 'logged_user' not in request.session:
        return redirect('login')
    
    user_email = request.session['logged_user']
    user = User_db.objects.get(email_db=user_email)
    
    try:
        csv_record = UserCSVFile.objects.get(user=user)
        
        if not csv_record.analysis_result:
            messages.error(request, "Please run analysis first")
            return redirect('advanced_analysis')
        
        # Read and preprocess CSV
        df = pd.read_csv(csv_record.file.path)
        df_processed = preprocess_bank_statement(df)
        
        # Create response with CSV
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="fraud_analysis_report_{csv_record.original_filename}"'
        
        # Write summary statistics
        writer = csv.writer(response)
        writer.writerow(['FRAUD ANALYSIS REPORT'])
        writer.writerow(['Generated on:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Original file:', csv_record.original_filename])
        writer.writerow([])
        
        # Write summary
        writer.writerow(['SUMMARY STATISTICS'])
        writer.writerow(['Total Transactions:', len(df_processed)])
        writer.writerow(['Fraud Probability:', f"{csv_record.analysis_result.get('fraud_probability', 0)*100:.2f}%"])
        writer.writerow(['High Risk Transactions:', csv_record.analysis_result.get('high_risk_count', 0)])
        writer.writerow(['Medium Risk Transactions:', csv_record.analysis_result.get('medium_risk_count', 0)])
        writer.writerow(['Total Amount at Risk:', f"${csv_record.analysis_result.get('total_amount_at_risk', 0):.2f}"])
        writer.writerow(['Average Risk Score:', f"{csv_record.analysis_result.get('risk_score', 0):.1f}/100"])
        writer.writerow([])
        
        # Write high-risk transactions
        writer.writerow(['HIGH RISK TRANSACTIONS (Risk Score >= 70)'])
        high_risk_df = df_processed[df_processed['Heuristic_Risk_Score'] >= 70]
        if not high_risk_df.empty:
            writer.writerow(['Date', 'Details', 'Amount', 'Risk Score', 'Fraud Flag'])
            for _, row in high_risk_df.iterrows():
                writer.writerow([
                    row['Date'] if pd.notna(row.get('Date')) else '',
                    str(row.get('Details', ''))[:100],  # Truncate long details
                    f"${row.get('Amount', 0):.2f}",
                    f"{row.get('Heuristic_Risk_Score', 0):.1f}",
                    row.get('Heuristic_Fraud_Flag', '')
                ])
        else:
            writer.writerow(['No high risk transactions found'])
        writer.writerow([])
        
        # Write all transactions with risk scores
        writer.writerow(['ALL TRANSACTIONS WITH RISK ASSESSMENT'])
        writer.writerow(['Date', 'Type', 'Amount', 'Balance', 'Risk Score', 'Fraud Flag', 'Transaction Channel'])
        
        for _, row in df_processed.iterrows():
            writer.writerow([
                row['Date'] if pd.notna(row.get('Date')) else '',
                row.get('Transaction_Type', ''),
                f"${row.get('Amount', 0):.2f}",
                f"${row.get('Balance', 0):.2f}",
                f"{row.get('Heuristic_Risk_Score', 0):.1f}",
                row.get('Heuristic_Fraud_Flag', ''),
                row.get('Transaction_Channel', '')
            ])
        
        return response
        
    except UserCSVFile.DoesNotExist:
        messages.error(request, "Please upload a CSV file first")
        return redirect('dashboard')
    except Exception as e:
        messages.error(request, f"Report generation error: {str(e)}")
        return redirect('advanced_analysis')


def get_analysis_data(request):
    """API endpoint to get analysis data for AJAX"""
    if 'logged_user' not in request.session:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    
    user_email = request.session['logged_user']
    user = User_db.objects.get(email_db=user_email)
    
    try:
        csv_record = UserCSVFile.objects.get(user=user)
        
        if not csv_record.analysis_result:
            return JsonResponse({'error': 'No analysis data available'}, status=404)
        
        return JsonResponse({
            'success': True,
            'analysis': csv_record.analysis_result
        })
        
    except UserCSVFile.DoesNotExist:
        return JsonResponse({'error': 'No CSV file found'}, status=404)


@require_POST
def analyze_csv_view(request):
    """Handle CSV analysis request"""
    if 'logged_user' not in request.session:
        return redirect('login')
    
    user_email = request.session['logged_user']
    user = User_db.objects.get(email_db=user_email)
    
    try:
        csv_record = UserCSVFile.objects.get(user=user)
        # Run the analysis
        return advanced_analysis_view(request)
        
    except UserCSVFile.DoesNotExist:
        messages.error(request, "Please upload a CSV file first")
        return redirect('dashboard')
    except Exception as e:
        messages.error(request, f"Analysis error: {str(e)}")
        return redirect('dashboard')


def debug_ml_status(request):
    """Debug page to check ML model status"""
    global ML_MODELS_LOADED, model, scaler
    
    model_dir = os.path.join(settings.BASE_DIR, "myapp", "ml")
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    context = {
        'ml_models_loaded': ML_MODELS_LOADED,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'model_exists': os.path.exists(model_path),
        'scaler_exists': os.path.exists(scaler_path),
        'model_dir': model_dir,
        'files_in_dir': os.listdir(model_dir) if os.path.exists(model_dir) else [],
    }
    
    return render(request, 'debug_ml.html', context)