# Login-and-Signup-using-Flask


This is a Flask admin dashboard template that displays user information in a tabular format.

![Login](screenshots/login.jpg)

## Installation

1. Clone the repository:
git clone https://github.com/Churanta/Login-and-Signup-using-Flask.git
2. Install all dependencies using pip3 install -r requirements.txt
3. Run python app.py to start server
4. Open http://localhost:5000 on your browser and you will see login page



## Usage

1. Open your web browser and go to `http://localhost:5000/admin/dashboard` to access the admin dashboard.

2. The admin dashboard will display a table with user information, including ID, username, photo, and mobile number.

3. The table rows have alternating colors to improve readability.

4. The user photo is displayed with a circular shape and a width of 100 pixels.

5. The actions column can be customized to include buttons or links for performing specific actions related to each user.

## Customization

You can customize the admin dashboard by modifying the templates and styles. Here's a brief overview of the relevant files:

- `templates/base.html`: The base HTML template that other templates extend from. You can modify the overall layout and structure of the admin dashboard here.

- `templates/admin_dashboard.html`: The template for the admin dashboard page. You can modify the table structure and styles here.

- `static/style.css`: The CSS file that contains styles for the admin dashboard. You can modify the styles to change the appearance of the dashboard.

- `static/photos/`: The folder where user photos are stored. You can place user photos in this folder and update the template accordingly.

Feel free to explore and customize the files to match your requirements.

## Screenshots

You can find screenshots of the admin dashboard below:

![login](screenshots/login.jpg)

![Admin Dashboard Screenshot 2](screenshots/s1.jpg)

![Signup](screenshots/signup.jpg)

## License

This project is licensed under the [MIT License](LICENSE).

