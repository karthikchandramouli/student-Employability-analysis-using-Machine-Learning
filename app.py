# app.py

from flask import Flask, render_template, request, redirect, url_for, jsonify,make_response
from flask_mysqldb import MySQL
from flask import session
import os
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'


app = Flask(__name__)


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'users'
app.config['MYSQL_PORT'] = 3306

mysql = MySQL(app)

# Base page
@app.route('/')
def base():
    return render_template('base.html')

# User signup
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Regular expression patterns for input validation
EMAIL_REGEX_PATTERN = r'^[\w\.-]+@[a-zA-Z\d\.-]+\.[a-zA-Z]{2,}$'
MOBILE_NUMBER_REGEX_PATTERN = r'^\d{10}$'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
UPLOAD_FOLDER = 'static/photos'
# Ensure the directory for storing user photos exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# User signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        photo = request.files['photo']
        mobile_number = request.form['mobile_number']

        # Check if passwords match
        if password != confirm_password:
            return jsonify({'message': 'Passwords do not match'})

        # Save user photo to a designated folder and get the file path
        photo_path = 'photos/' + photo.filename
        photo.save(photo_path)

        # Insert user data into the database
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password, photo, mobile_number) VALUES (%s, %s, %s, %s, %s)",
            (username, email, password, photo_path, mobile_number))
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('login'))

    return render_template('signup.html')

# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if email and password match in the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cur.fetchone()
        cur.close()

        if user:
            # Set session variables or perform any desired action for successful login
            return redirect(url_for('home'))
        else:
            return jsonify({'message': 'Invalid email or password'})

    return render_template('login.html')

# Generate a secret key
secret_key = os.urandom(24)
app.secret_key = secret_key

# Set up MinMaxScaler
scaler = MinMaxScaler()

# Function to fetch Excel files for a selected year
@app.route('/home/get_files_excel')
def get_files_excel():
    year = request.args.get('year')
    folder_path = os.path.join(app.config["UPLOAD_FOLDER"], year)
    excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    return jsonify(excel_files)

# Function to get the list of year folders
def get_year_folders():
    return [folder for folder in os.listdir(app.config["UPLOAD_FOLDER"]) if os.path.isdir(os.path.join(app.config["UPLOAD_FOLDER"], folder))]

@app.route('/home/load_excel', methods=['POST'])
def load_excel():
    excel_file_path = request.form['file_path']
    
    # Load the Excel file into the DataFrame df
    df = pd.read_excel(excel_file_path)
    
    # Encode the internship categories into integers
    internship_mapping = {'CLOUD': 1, 'WEB_DEVELOPMENT': 2, 'DEVOPS': 3, 'CYBER_SECURITY': 4}
    df['INTERNSHIP'] = df['INTERNSHIP'].map(internship_mapping)
    
    # Normalize the DSA, DBMS, CNS, CGPA, and Internship values using Min-Max scaling
    df[['DSA', 'DBMS', 'CNS', 'CGPA', 'INTERNSHIP']] = scaler.fit_transform(df[['DSA', 'DBMS', 'CNS', 'CGPA', 'INTERNSHIP']])
    
    # Convert DataFrame to dictionary and then to JSON
    df_json = df.to_json(orient='records')
    
    # Store the JSON data in a session variable for later use in the 'home' route
    session['df'] = df_json
    
    return '', 204

@app.route('/home', methods=['GET', 'POST'])
def home():
    # Get the list of year folders
    year_folders = get_year_folders()
    
    if request.method == 'POST':
        # Get user-entered priorities for each field
        priorities = {}
        for i in range(1, 6):
            field = request.form.get(f'priority{i}_SelectedOptions')
            priorities[field] = float(request.form.get(f'priority{i}_selected'))

        # Assign fixed priority values from priority 1 to 5
        priority_values = []
        for field in ['CGPA', 'INTERNSHIP', 'DSA', 'CNS', 'DBMS']:
            priority_values.append(priorities[field])

        # Retrieve the JSON data from the session and convert it back to DataFrame
        df_json = session.get('df', '[]')
        df = pd.read_json(io.StringIO(df_json), orient='records')

        if not df.empty:
            # Calculate the weighted scores for each student using fixed priorities
            df['WEIGHTED_SCORE'] = df[['CGPA', 'INTERNSHIP', 'DSA', 'CNS', 'DBMS']].dot(priority_values)

            # Initialize and train the Random Forest model
            X = df[['DSA', 'DBMS', 'CNS', 'CGPA', 'INTERNSHIP']]
            y = df['WEIGHTED_SCORE']
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)

            # Predict the weighted scores for all students using the Random Forest model
            df['PREDICTED_WEIGHTED_SCORE'] = rf_model.predict(X)

            # Sort the DataFrame based on predicted weighted scores
            sorted_df_rf = df.sort_values(by='PREDICTED_WEIGHTED_SCORE', ascending=False)
            # Inverse transform the actual scores
            actual_scores = scaler.inverse_transform(sorted_df_rf[['DSA', 'DBMS', 'CNS', 'CGPA', 'INTERNSHIP']].values)
        
            # Add the actual scores to the DataFrame
            sorted_df_rf[['DSA_ACTUAL', 'DBMS_ACTUAL', 'CNS_ACTUAL', 'CGPA_ACTUAL', 'INTERNSHIP_ACTUAL']] = actual_scores

            # Get the number of top students specified by the user
            num_top_students = int(request.form.get('top_students'))

            # Select the top students based on predicted weighted scores
            top_students_rf_final = sorted_df_rf.head(num_top_students)

            # Drop the columns CGPA, DSA, CNS, DBMS, and INTERNSHIP before rendering the HTML table
            html_table = top_students_rf_final.drop(columns=['CGPA', 'DSA', 'CNS', 'DBMS', 'INTERNSHIP']).to_html(index=False)
            
            return render_template('home.html', year_folders=year_folders, table=html_table, error_message=None)

    # If request method is GET or DataFrame is empty, render the form without data
    return render_template('home.html', year_folders=year_folders, table=None, error_message=None)


# Define route to download PDF
@app.route('/home/download_csv', methods=['POST'])
def download_csv():
    html_content = request.form.get('htmlContent', '')

    # Convert HTML content to pandas DataFrame
    df = pd.read_html(html_content)[0]

    # Generate CSV file from DataFrame
    csv_output = df.to_csv(index=False)

    # Set up response to return the CSV file
    response = make_response(csv_output)
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=top_students.csv'
    return response

# Admin login
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if admin email and password match
        if email == 'admin@gmail.com' and password == 'admin':
            # Set session variables or perform any desired action for successful admin login
            return redirect(url_for('admin_dashboard'))
        else:
            error_message = 'Invalid email or password'
            return render_template('admin_login.html', error=error_message)

    # Render the admin login page for GET requests
    return render_template('admin_login.html')



# Admin dashboard route
@app.route('/admin/dashboard')
def admin_dashboard():
    # Fetch all user details from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    cur.close()

    # Convert fetched tuples to dictionaries
    users_dict = []
    for user in users:
        user_dict = {
            'id': user[0],
            'username': user[1],
            'mobile_number': user[5]
        }
        users_dict.append(user_dict)

    return render_template('admin_dashboard.html', users=users_dict)

# Student details
app.config["UPLOAD_FOLDER"] = "static/excel"


@app.route("/student-details", methods=['GET', 'POST'])
def student_details():
    batch_year = None
    added_batches = []

    if request.method == "POST":
        # Validate batch and upload_file fields
        batch_year = request.form.get('batch')
        upload_file = request.files.get('upload_file')

        if not batch_year:
            return render_template("error.html", message="Batch year is required."), 400

        if not upload_file:
            return render_template("error.html", message="No file uploaded."), 400

        if upload_file.filename == '':
            return render_template("error.html", message="Please select a file to upload."), 400

        # Check if upload_file has the correct file extension
        if not upload_file.filename.endswith('.xlsx'):
            return render_template("error.html", message="Invalid file type. Please upload an Excel file."), 400

        batch_folder = os.path.join(app.config["UPLOAD_FOLDER"], str(batch_year))
        if not os.path.exists(batch_folder):
            os.makedirs(batch_folder)

        # Update file path to include batch year folder
        filename = secure_filename(upload_file.filename)
        filepath = os.path.join(batch_folder, filename)
        upload_file.save(filepath)

        # Read Excel file
        try:
            data = pd.read_excel(filepath)
            html_content = data.to_html(index=False).replace('<th>', '<th style="text-align:center">')

            # Get added batches
            for key, value in request.form.items():
                if key.startswith("added_batch_"):
                    added_batches.append(value)

            return render_template("student_details.html", data=html_content, batch_year=batch_year, added_batches=added_batches)
        except Exception as e:
            return render_template("error.html", message=f"Error reading Excel file: {str(e)}"), 500

    elif request.method == "GET":
        # Get added batches from session if available
        added_batches = session.get("added_batches", [])

    return render_template("student_details.html", batch_year=batch_year, added_batches=added_batches)



#Reports page 
@app.route('/reports')
def reports():
    # Get the list of year folders containing Excel files
    year_folders = [folder for folder in os.listdir(app.config["UPLOAD_FOLDER"]) if os.path.isdir(os.path.join(app.config["UPLOAD_FOLDER"], folder))]
    return render_template('reports.html', year_folders=year_folders)

@app.route('/reports/get_excel_files')
def get_excel_files():
    year = request.args.get('year')
    folder_path = os.path.join(app.config["UPLOAD_FOLDER"], year)
    excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    return jsonify(excel_files)

@app.route('/reports/display_excel')
def display_excel():
    # Get the file path from the request arguments
    excel_file_path = request.args.get('file_path')
    
    # Check if the file path is provided
    if not excel_file_path:
        return jsonify({"error": "File path parameter is missing."}), 400
    
    try:
        # Read the Excel file using pandas
        data = pd.read_excel(excel_file_path)
        
        # Convert the data to HTML format
        html_content = data.to_html(index=False)
        return html_content
    
    except FileNotFoundError:
        return jsonify({"error": "File not found."}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/datavisualization")
def datavisualization():
    year_folders = [folder for folder in os.listdir(app.config["UPLOAD_FOLDER"]) if os.path.isdir(os.path.join(app.config["UPLOAD_FOLDER"], folder))]
    return render_template('datavisualization.html', year_folders=year_folders)

@app.route('/datavisualization/get_excel_files')
def get_excel_files_route():
    year = request.args.get('year')
    folder_path = os.path.join(app.config["UPLOAD_FOLDER"], year)
    excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    return jsonify(excel_files)

@app.route('/datavisualization/display_visualization')
def display_visualization():
    excel_file_path = request.args.get('file_path')

    if not os.path.exists(excel_file_path):
        return jsonify({"error": "Invalid file path."}), 400

    try:
        with open(excel_file_path, 'rb') as f:
            f.read()
    except Exception as e:
        return jsonify({"error": "Unable to read file."}), 400

    chart_paths = generate_charts(excel_file_path)
    return jsonify(chart_paths)

def generate_charts(excel_file_path):
    try:
        data = pd.read_excel(excel_file_path)
        required_columns = ['BRANCH', 'CGPA', 'LEET_RANK', 'DSA', 'CNS', 'DBMS', 'INTERNSHIP']
        if not all(column in data.columns for column in required_columns):
            raise ValueError("Missing required columns in Excel file.")

        excel_file_name = os.path.basename(excel_file_path)
        histogram_path = generate_branch_cgpa_histogram(data, excel_file_name)
        box_plot_path = generate_box_plot(data, excel_file_name)
        pie_chart_paths = generate_internship_pie_chart(data, excel_file_name)
        scatter_cgpa_vs_cet_rank_path = generate_scatter_cgpa_vs_cet_rank(data, excel_file_name)
        grouped_bar_chart_subject_performance_path = generate_grouped_bar_chart_subject_performance(data, excel_file_name)
        pie_chart_internship_domain_preference_path = generate_pie_chart_internship_domain_preference(data, excel_file_name)
        scatter_cgpa_vs_internship_domain_path = generate_scatter_cgpa_vs_internship_domain(data, excel_file_name)
        heatmap_path = generate_heatmap(data, excel_file_name)

        return {
            "Branch-wise CGPA Distribution": histogram_path,
            "Subject-wise Performance Comparison": box_plot_path,
            "Internship Domain Distribution": pie_chart_paths,
            "CGPA vs. LEET_RANK": scatter_cgpa_vs_cet_rank_path,
            "Branch-wise Subject Performance": grouped_bar_chart_subject_performance_path,
            "Internship Domain Preference": pie_chart_internship_domain_preference_path,
            "CGPA vs. Internship Domain": scatter_cgpa_vs_internship_domain_path,
            "Correlation Heatmap": heatmap_path
        }
    
    except Exception as e:
        return {"error": str(e)}

def generate_branch_cgpa_histogram(data, excel_file_name):
    try:
        if 'BRANCH' not in data.columns or 'CGPA' not in data.columns:
            raise ValueError("Required columns 'BRANCH' or 'CGPA' are missing in the data.")
        
        plt.figure(figsize=(10, 6))
        for branch in data['BRANCH'].unique():
            branch_data = data[data['BRANCH'] == branch]['CGPA']
            plt.hist(branch_data, alpha=0.5, label=branch)
        
        plt.xlabel('CGPA')
        plt.ylabel('Frequency')
        plt.title('Branch-wise CGPA Distribution')
        plt.legend()
        
        histogram_path = f"static/branch_cgpa_histogram_{excel_file_name}.png"
        plt.savefig(histogram_path, bbox_inches='tight')
        plt.close()
        
        return histogram_path
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def generate_box_plot(data, excel_file_name):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        boxplot_data = data[['DSA', 'CNS', 'DBMS']]
        plt.figure(figsize=(10, 6))
        boxplot_data.boxplot()
        
        plt.title('Subject-wise Performance Comparison')
        plt.ylabel('Marks')
        plt.xlabel('Subjects')
        
        box_plot_path = f'static/box_plot_{excel_file_name}.png'
        plt.savefig(box_plot_path, bbox_inches='tight')
        plt.close()
        
        return box_plot_path
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def generate_internship_pie_chart(data, excel_file_name):
    try:
        if not isinstance(data, pd.DataFrame) or 'BRANCH' not in data.columns or 'INTERNSHIP' not in data.columns:
            raise ValueError("Input data must be a pandas DataFrame with 'BRANCH' and 'INTERNSHIP' columns.")
        
        branch_internship_counts = data.groupby('BRANCH')['INTERNSHIP'].value_counts().unstack(fill_value=0)
        pie_chart_paths = []
        
        for branch in branch_internship_counts.index:
            branch_data = branch_internship_counts.loc[branch]
            plt.figure(figsize=(8, 8))
            branch_data.plot(kind='pie', autopct='%1.1f%%', startangle=90)
            plt.title(f'Internship Domain Distribution for {branch}')
            plt.ylabel('')
            plt.legend(title='Internship Domain', loc='best')
            plt.tight_layout()
            
            pie_chart_path = f'static/internship_pie_chart_{branch}_{excel_file_name}.png'
            plt.savefig(pie_chart_path)
            plt.close()
            
            pie_chart_paths.append(pie_chart_path)
        
        return pie_chart_paths
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def generate_scatter_cgpa_vs_cet_rank(data, excel_file_name):
    try:
        if not isinstance(data, pd.DataFrame) or 'LEET_RANK' not in data.columns or 'CGPA' not in data.columns:
            raise ValueError("Input data must be a pandas DataFrame with 'LEET_RANK' and 'CGPA' columns.")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(data['LEET_RANK'], data['CGPA'])
        plt.xlabel('LEET_RANK')
        plt.ylabel('CGPA')
        plt.title('CGPA vs. LEET_RANK')
        scatter_cgpa_vs_cet_rank_path = f'static/scatter_cgpa_vs_cet_rank_{excel_file_name}.png'
        plt.savefig(scatter_cgpa_vs_cet_rank_path, bbox_inches='tight')
        plt.close()
        return scatter_cgpa_vs_cet_rank_path
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def generate_grouped_bar_chart_subject_performance(data, excel_file_name):
    try:
        if not isinstance(data, pd.DataFrame) or 'BRANCH' not in data.columns or all(col not in data.columns for col in ['DSA', 'CNS', 'DBMS']):
            raise ValueError("Input data must be a pandas DataFrame with 'BRANCH' and at least one of ['DSA', 'CNS', 'DBMS'] columns.")
        
        subject_columns = ['DSA', 'CNS', 'DBMS']
        BRANCH_subject_means = data.groupby('BRANCH')[subject_columns].mean()
        plt.figure(figsize=(10, 6))
        BRANCH_subject_means.plot(kind='bar')
        plt.title('Branch-wise Subject Performance')
        plt.ylabel('Mean Marks')
        plt.xlabel('BRANCH')
        grouped_bar_chart_subject_performance_path = f'static/grouped_bar_chart_subject_performance_{excel_file_name}.png'
        plt.savefig(grouped_bar_chart_subject_performance_path, bbox_inches='tight')
        plt.close()
        return grouped_bar_chart_subject_performance_path
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def generate_pie_chart_internship_domain_preference(data, excel_file_name):
    try:
        if not isinstance(data, pd.DataFrame) or 'INTERNSHIP' not in data.columns:
            raise ValueError("Input data must be a pandas DataFrame with 'INTERNSHIP' column.")
        
        plt.figure(figsize=(10, 6))
        data['INTERNSHIP'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Internship Domain Preference')
        plt.ylabel('')
        pie_chart_internship_domain_preference_path = f'static/pie_chart_internship_domain_preference_{excel_file_name}.png'
        plt.savefig(pie_chart_internship_domain_preference_path, bbox_inches='tight')
        plt.close()
        return pie_chart_internship_domain_preference_path
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def generate_scatter_cgpa_vs_internship_domain(data, excel_file_name):
    try:
        if not isinstance(data, pd.DataFrame) or 'INTERNSHIP' not in data.columns or 'CGPA' not in data.columns:
            raise ValueError("Input data must be a pandas DataFrame with 'INTERNSHIP' and 'CGPA' columns.")
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data['INTERNSHIP'], y=data['CGPA'])
        plt.xticks(rotation=90)
        plt.title('CGPA vs. Internship Domain')
        plt.xlabel('Internship Domain')
        plt.ylabel('CGPA')
        scatter_cgpa_vs_internship_domain_path = f'static/scatter_cgpa_vs_internship_domain_{excel_file_name}.png'
        plt.savefig(scatter_cgpa_vs_internship_domain_path, bbox_inches='tight')
        plt.close()
        return scatter_cgpa_vs_internship_domain_path
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def generate_heatmap(data, excel_file_name):
    try:
        heatmap_data = data[['CGPA', 'LEET_RANK', 'DSA', 'CNS', 'DBMS']]
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        heatmap_path = f'static/heatmap_{excel_file_name}.png'
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
        return heatmap_path
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
# Logout
@app.route('/logout')
def logout():
    # Perform logout actions, such as clearing session variables
    # Redirect to the login page or any other desired page
    return redirect(url_for('base'))

@app.route('/adminlogout')
def adminlogout():
    return redirect(url_for('base'))

# Run the application
if __name__ == '__main__':
    app.debug = False
    app.run()
