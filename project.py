"""
Senior Well-Being Analysis Tool: A comprehensive tool for analyzing and visualizing senior well-being data,
as well as featuring GenWell 2024 Cross-Sectional Data collection, analysis, visualization, and benchmarking capabilities.

Features:
- Data preprocessing and cleaning
- Statistical analysis
- Benchmarking
- Data visualization
- GUI-based user data collection
- Database storage, protection, retrieval
- Reporting user data

Dependencies:
- pandas
- matplotlib
- seaborn
- turtle (built-in)
- sqlite3 (built-in)

Author: Nayeon Kang, Xinyue (Jasmine) Zhou, & Yifan Wang
Version: 1.0.0
Code Last Updated: 2024-12-02
"""

# The following libraries are imported to support data cleaning, statistical analysis, and interactive visualizations:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import turtle
import hashlib # The hashlib module provides secure hash functions such as SHA-256
import sqlite3 # SQLite3 allows for interaction with SQLite databases directly from Python
import os
from datetime import datetime, timedelta # The datetime module provides classes for working with dates and times


# Module 1: Preprocess and Categorize Data
# This module is responsible for preprocessing and categorizing data, particularly for handling survey responses related to social time, physical health, mental health, face-to-face interactions, and volunteering.
def preprocess_categorical_data(df):
    """
    Preprocess and categorize responses for analysis and visualization.

    Args:
        df (pd.DataFrame): Raw input DataFrame containing survey responses
            Required columns:
            - CONNECTION_social_time_alone
            - WELLNESS_self_rated_physical_health
            - WELLNESS_self_rated_mental_health
            - CONNECTION_activities_face_to_face_convorsation_p3m
            - CONNECTION_activities_volunteered_p3m
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with mapped categorical values

    Raises:
        KeyError: If required columns are missing
        ValueError: If data contains invalid values
        
    """        

    # Ensure 'CONNECTION_social_time_alone' is numeric
    # Ensures that non-numeric values are converted to NaN, which are then removed    
    df.loc[:, "CONNECTION_social_time_alone"] = pd.to_numeric(
        df["CONNECTION_social_time_alone"], errors="coerce"
    )
    df = df.dropna(subset=["CONNECTION_social_time_alone"])


    # Categorize 'CONNECTION_social_time_alone' into defined hourly ranges 
    # These ranges represent weekly hours spent alone, aiding in grouping data.    
    df.loc[:, "CONNECTION_social_time_alone"] = pd.cut(
        df["CONNECTION_social_time_alone"],
        # Map social time categories to numeric values for analysis        
        bins=[0, 20, 40, 80, 120, 168],   # Weekly hours are divided into ranges
        labels=["0-20", "21-40", "41-80", "81-120", "121-168"],
        include_lowest=True
    )
    
    
    # Map self-rated physical health responses to numeric values --> To facilitate statistical comparisons
    df.loc[:, "WELLNESS_self_rated_physical_health"] = (
        df["WELLNESS_self_rated_physical_health"]
        .astype(str)  # Convert to string for consistent processing
        .str.strip()  # Remove leading/trailing spaces
        .replace({
            "Poor": 1,
            "Fair": 2,
            "Good": 3,
            "Very good": 4,
            "Excellent": 5
        })
    )


    # Map self-rated mental health responses to numeric values
    df.loc[:, "WELLNESS_self_rated_mental_health"] = (
        df["WELLNESS_self_rated_mental_health"]
        .astype(str)  # Convert to string for consistent processing
        .str.strip()  # Remove leading/trailing spaces
        .replace({
            "Poor": 1,
            "Fair": 2,
            "Good": 3,
            "Very good": 4,
            "Excellent": 5
        })
    )
  
    # Map face-to-face conversation frequency responses to numeric values
    df.loc[:, "CONNECTION_activities_face_to_face_convorsation_p3m"] = (
        df["CONNECTION_activities_face_to_face_convorsation_p3m"]
        .astype(str)  # Convert to string for consistent processing
        .str.strip()  # Remove leading/trailing spaces
        .replace({
            "Not in the past three months": 0,
            "Less than monthly": 1,
            "Monthly": 2,
            "A few times a month": 3,
            "Weekly": 4,
            "A few times a week": 5,
            "Daily or almost daily": 6
        })
    )
    
    # Map volunteering frequency responses to numerical values
    df.loc[:, "CONNECTION_activities_volunteered_p3m"] = (
        df["CONNECTION_activities_volunteered_p3m"]      
        .astype(str)
        .str.strip()    
        .replace({
            "Not in the past three months": 0,
            "Less than monthly": 1,
            "Monthly": 2,
            "A few times a month": 3,
            "Weekly": 4,
            "A few times a week": 5,            
            "Daily or almost daily": 6
        })
    )

    return df



# Module 2: Load & Clean Data
# This module is designed to:
# - Load raw data from a CSV file.
# - Filter and retain specific columns for analysis
# - Clean data by handling missing values
def load_and_clean_data(file_path):
    """
    Load and clean the dataset from a csv file (2024 Cross-Sectional Data.csv)

    Args:
        file_path (str): Path to the CSV dataset file

    Returns:
        pd.DataFrame: A cleaned DataFrame with:
                      - Retained relevant columns
                      - Filtered rows for participants aged 65 or above
                      - Processed categorical variables

    """
    # Load the dataset into a DataFrame
    df = pd.read_csv(file_path)
    
    # To retain only the required columns for analysis
    columns_needed = [
        'PARTICIPANT_ID',
        'DEMO_age',
        'LONELY_dejong_emotional_social_loneliness_scale_TOTAL',
        'CONNECTION_social_time_alone',
        'CONNECTION_activities_face_to_face_convorsation_p3m',
        'WELLNESS_self_rated_physical_health',
        'WELLNESS_self_rated_mental_health',
        'CONNECTION_activities_volunteered_p3m'
    ]
    
    # Keep only the specified columns
    df = df[columns_needed].astype(str)  # Convert all columns to strings to handle mixed types
    
    # Preprocess and categorize relevant columes (aka variables)
    df = preprocess_categorical_data(df)        
    
    # Convert 'DEMO_age' column to numeric for filtering by age
    df['DEMO_age'] = pd.to_numeric(df['DEMO_age'], errors='coerce')
    
    # Filter for participants aged 65+
    df = df[df['DEMO_age'] >= 65]    
    
    # Replace missing values ('9999') with NaN and drop rows with missing values
    df = df.replace('9999', pd.NA).dropna() 

    return df


# Module 3: Visualization Functions
# This module provides functionality to generate various visualizations that explore:
# - The relationship between loneliness and time spent alone
# - Correlations between loneliness, physical health, and mental health
# - The impact of social interactions (face-to-face conversations and volunteering) on loneliness scores
def generate_visualizations(df):
    """
    Generate graphs comparing Loneliness Scale Total with other variables.
    
    Visualization Menu:
    1. Box Plot: Loneliness vs. Time Spent Alone
    2. Heatmap: Correlation between Loneliness, Physical, and Mental Health
    3. Bar Chart: Loneliness vs. Face-to-Face Conversation Frequency
    4. Bar Chart: Loneliness vs. Volunteering Frequency

    The user interacts with a menu-driven interface to select a visualization.

    Args:
        df (pd.DataFrame): Preprocessed and cleaned dataset ready for visualization.
        
    """
    while True:
        print("\nVisualization Menu Part 1:")
        print("1. Box & Whisker Plot: Loneliness vs. Time Spent Alone (Categorized)")
        print("2. Heatmap: Loneliness vs. Physical and Mental Health")
        print("3. Bar Chart: Loneliness vs. Face-to-Face Conversations")
        print("4. Bar Chart: Loneliness vs. Volunteering Frequency")
        print("5. Exit Visualizations")
  
        choice = input("Select an option (1-5): ")

        if choice == '1':
            # Box Plot: Loneliness vs. Time Spent Alone            
            plt.figure(figsize=(8, 6)) # Set the figure size for the visualization
            # Ensure the categories are ordered explicitly
            df["CONNECTION_social_time_alone"] = pd.Categorical(
                df["CONNECTION_social_time_alone"],  
                categories=["0-20", "21-40", "41-80", "81-120", "121-168"],
                ordered=True
            )            
            sns.boxplot(
                x="CONNECTION_social_time_alone",
                y="LONELY_dejong_emotional_social_loneliness_scale_TOTAL",
                data=df,
                order=["0-20", "21-40", "41-80", "81-120", "121-168"]  # Explicitly set order here as well because: Even if the pd.Categorical ensures the correct order in the data, sns.boxplot does not rely on the underlying categorical order. By default, Seaborn ordered categories alphabetically or in our case randomly based on the order they appear in the data.
            )
            plt.title("Loneliness Score vs. Time Spent Alone")
            plt.xlabel("Time Spent Alone (Hours)")
            plt.ylabel("Loneliness Score")
            plt.tight_layout()
            plt.show()  
        elif choice == '2':
            # Heatmap: Loneliness vs. Physical and Mental Health
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                df[[
                    "LONELY_dejong_emotional_social_loneliness_scale_TOTAL",
                    "WELLNESS_self_rated_physical_health",
                    "WELLNESS_self_rated_mental_health"
                ]].corr(),
                annot=True, cmap="coolwarm", square=True
            )
            plt.title("Correlation: Loneliness, Physical and Mental Health")
            plt.show()
        elif choice == '3':
            # Bar Chart: Loneliness vs. Face-to-Face Conversations   
            plt.figure(figsize=(8, 6))
            
            # Define the order of the categories for the x-axis
            # Explicitly define the order of interaction frequencies to ensure consistency in visualization            
            category_order = [
                "Not in the past three months",
                "Less than monthly",
                "Monthly",
                "A few times a month",
                "Weekly",
                "A few times a week",
                "Daily or almost daily"
            ]                        
            
            # Map numeric scale back to labels for better visualization
            # Translate numeric values (0-6) into descriptive labels for clarity in the chart            
            label_mapping = {
                0: "Not in the past three months",
                1: "Less than monthly",
                2: "Monthly",
                3: "A few times a month",
                4: "Weekly",
                5: "A few times a week",
                6: "Daily or almost daily"     
            }
            
            # Replace numeric values with corresponding labels
            df["CONNECTION_activities_face_to_face_convorsation_p3m"] = (
                df["CONNECTION_activities_face_to_face_convorsation_p3m"]
                .replace(label_mapping))
            
            # Ensure the 'Loneliness Score' column is numeric
            df["LONELY_dejong_emotional_social_loneliness_scale_TOTAL"] = pd.to_numeric(
                df["LONELY_dejong_emotional_social_loneliness_scale_TOTAL"], errors='coerce')
        
            # Create the cleaned DataFrame
            # Remove rows with missing values to ensure accurate plotting            
            df_cleaned = df.dropna(subset=[
                "CONNECTION_activities_face_to_face_convorsation_p3m",
                "LONELY_dejong_emotional_social_loneliness_scale_TOTAL"
            ])
            
            # Create the bar chart with the specified order
            sns.barplot(
                x="CONNECTION_activities_face_to_face_convorsation_p3m",
                y="LONELY_dejong_emotional_social_loneliness_scale_TOTAL",
                data=df_cleaned,
                ci=None, # Disable confidence intervals for simplicity
                palette="Blues_d",
                order=category_order  # Ensure correct order of the x-axis categories
            )
                              
            # Set the title and labels  
            # Clearly label the chart to convey its meaning to the users          
            plt.title("Loneliness Score vs. Face-to-Face Conversations")
            plt.xlabel("Frequency of Face-to-Face Conversations")
            plt.ylabel("Loneliness Score")
            
            # Set y-axis limits based on the range of your data
            # Generated by ChatGPT: Dynamically adjust the upper limit of the y-axis by adding 1 to the maximum score. Without the buffer, the highest bar in the plot was touching & overlap with the top edge of the plot.        
            plt.ylim(0, df_cleaned["LONELY_dejong_emotional_social_loneliness_scale_TOTAL"].max() + 1)
            
            # Generated by ChatGPT: Rotate x-axis labels for better readability --> This is to ensure long labels do not overlap by rotating them
            plt.xticks(rotation=45, ha="right")  
            
            # Display the plot   
            # Adjust layout for a clean look and show the plot            
            plt.tight_layout()            
            plt.show()
            
        elif choice == '4':
            # Bar Plot: Loneliness vs. Volunteering Frequency
            plt.figure(figsize=(8, 6))
            
            # Group the data by volunteering frequency and calculate mean and standard deviation            
            grouped = df.groupby("CONNECTION_activities_volunteered_p3m")["LONELY_dejong_emotional_social_loneliness_scale_TOTAL"]
            means = grouped.mean()
            stds = grouped.std()

            # Map numeric scale back to labels for better visualization
            label_mapping = {
                0: "Not in the past three months",
                1: "Less than monthly",
                2: "Monthly",
                3: "A few times a month",
                4: "Weekly",
                5: "A few times a week",
                6: "Daily or almost daily"
            }
            means.index = means.index.map(label_mapping)  # Replace numeric indices with labels

            # Plot the bar chart with error bars
            means.plot(kind='bar', yerr=stds, capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
            
            # Set the title and labels to provide meaningful context to the chart
            plt.title("Average Loneliness Score by Volunteering Frequency")
            plt.xlabel("Volunteering Frequency (Last 3 Months)")
            plt.ylabel("Average Loneliness Score")
            
            # Generated by ChatGPT: Rotate x-axis labels for better readability --> This is to ensure long labels do not overlap by rotating them            
            plt.xticks(rotation=45, ha="right")
            
            # Adjust layout for clean appearance            
            plt.tight_layout()
            plt.show()
        elif choice == '5':
            # Exit the visualization menu           
            print("Exiting Visualizations.")
            break
        else:
            # Handle invalid menu choices            
            print("Invalid choice. Please select a valid option.")

# Module 4: Compute Benchmarks from the CSV File
def calculate_benchmarks(df):
    """
    Calculate benchmarks (mean, median) for relevant variables from the dataset
    
    Parameters:
        df (pd.DataFrame): The cleaned dataset with preprocessed and categorized variables
    
    Returns:
        dict: A dictionary containing mean and median benchmarks for each variable of interest.

    Returns:
        dict: Benchmarks for relevant variables.
    """
    # Map "CONNECTION_social_time_alone" into categorical values 0-4
    bins = [0, 20, 40, 80, 120, 168]  # Weekly hour ranges for categorization
    labels = [0, 1, 2, 3, 4]  # Numeric labels representing each range
    df["CONNECTION_social_time_alone"] = pd.cut(
        df["CONNECTION_social_time_alone"], 
        bins=bins, 
        labels=labels, 
        include_lowest=True) # Ensure that the lowest value (0) is included in the first bin
    
    # Convert the column to numeric for calculation purposes
    # This step is necessary because the categorization converts the column to a categorical data type    
    df["CONNECTION_social_time_alone"] = df["CONNECTION_social_time_alone"].astype(float)

    # Initialize a dictionary to store mean and median benchmarks for key variables
    benchmarks = {
        "CONNECTION_social_time_alone": {
            "mean": df["CONNECTION_social_time_alone"].mean(),
            "median": df["CONNECTION_social_time_alone"].median(),
        },
        "WELLNESS_self_rated_physical_health": {
            "mean": df["WELLNESS_self_rated_physical_health"].mean(),
            "median": df["WELLNESS_self_rated_physical_health"].median(),
        },
        "WELLNESS_self_rated_mental_health": {
            "mean": df["WELLNESS_self_rated_mental_health"].mean(),
            "median": df["WELLNESS_self_rated_mental_health"].median(),
        },
        "CONNECTION_activities_face_to_face_convorsation_p3m": {
            "mean": df["CONNECTION_activities_face_to_face_convorsation_p3m"].mean(),
            "median": df["CONNECTION_activities_face_to_face_convorsation_p3m"].median(),  
        },
        "CONNECTION_activities_volunteered_p3m": {
            "mean": df["CONNECTION_activities_volunteered_p3m"].mean(),
            "median": df["CONNECTION_activities_volunteered_p3m"].median(),
        },
    }
    
    # Return the dictionary containing calculated benchmarks    
    return benchmarks

# Function to save benchmarks to a file
def save_benchmarks_to_file(benchmarks, file_name="benchmarks.txt"):
    """
    Save benchmark results to a file in a user-friendly format.
    
    Parameters:
        benchmarks (dict): Dictionary containing benchmark results for variables in the dataset.
                           Each variable has mean and median values as keys.
        file_name (str): Name of the file to save the benchmarks.
                         Defaults to "benchmarks.txt".

    Output:
        A text file containing formatted benchmark results.
        
    """
    # Open the specified file in write mode    
    with open(file_name, "w") as file:
        # Writing a header line to introduce the content of the file        
        file.write("Benchmarks for the dataset:\n\n")
        # Loop through the benchmark dictionary to format and write each variable's stats        
        for variable, stats in benchmarks.items():
            # Write the variable name            
            file.write(f"{variable}:\n")
            # Write the mean value rounded to 2 decimal places            
            file.write(f"  Mean: {stats['mean']:.2f}\n")
            # Write the median value rounded to 2 decimal places            
            file.write(f"  Median: {stats['median']:.2f}\n\n")
    print(f"Benchmarks are successfully saved to {file_name}.")  # Print a confirmation message after saving the file

# Example usage of the script. We need this because if there are issues (which there was before) with the script or the functions, this block provides an isolated way to test and debug them
if __name__ == "__main__":
    # Replace this with our actual cleaned DataFrame
    df = pd.DataFrame({
        "LONELY_dejong_emotional_social_loneliness_scale_TOTAL": [3, 4, 5, 4],
        "CONNECTION_activities_face_to_face_convorsation_p3m": [1, 2, 3, 4],
        "WELLNESS_self_rated_physical_health": [3, 4, 5, 3],
        "WELLNESS_self_rated_mental_health": [4, 5, 3, 4],
        "CONNECTION_social_time_alone": [10, 35, 90, 150],  # Example data
        "CONNECTION_activities_volunteered_p3m": [0, 1, 2, 3],
    })

    # Call the calculate_benchmarks function to compute the benchmarks for the dataset
    benchmarks = calculate_benchmarks(df)
    
    # Save the benchmarks to a file for reporting    
    save_benchmarks_to_file(benchmarks)


# Module 5: Turtle-Based Data Entry
# This module allows users to input survey data through a graphical interface using the Turtle library
# It includes functionalities such as resetting data, anonymizing user details, checking question intervals, and saving responses to a database.
def turtle_based_data_entry():
    """
    Collect user survey data using a Turtle-based graphical interface with proper time intervals for questions.
    """  
    
    # Function to reset survey data    
    def reset_survey_data():
        """
        Delete all existing survey data files and database
        
        Ensures a fresh start for data collection by removing previous entries

        """
        if os.path.exists("all_user_data.txt"):
            try:
                os.remove("all_user_data.txt")
                print("all_user_data.txt has been deleted.")
            except Exception as e:
                print(f"Error deleting all_user_data.txt: {e}")
        
        if os.path.exists("survey_data.db"):
            try:
                os.remove("survey_data.db")
                print("survey_data.db has been deleted.")
            except Exception as e:
                print(f"Error deleting survey_data.db: {e}")

    # Function to anonymize user names
    def anonymize_name(name):
        """
        Anonymize user name using SHA-256 hashing
        
        Args: name (str): User's name to anonymize
        
        Returns:str: SHA-256 hash of the input name
        
        Example: anonymize_name("Nayeon")
        '7ab...'
       
        """
        return hashlib.sha256(name.encode()).hexdigest()
    
    # Function to set up the survey database    
    def setup_database():
        """
        Connect to the SQLite database for storing survey responses and intervals
                
        - Creates tables `survey_responses` and `question_intervals` if they do not exist
        - Returns:
            conn (sqlite3.Connection): Connection object for the database
            cursor (sqlite3.Cursor): Cursor object for executing SQL queries
                       
        """
        conn = sqlite3.connect("survey_data.db")
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            survey_date TEXT,
            question TEXT,
            response INTEGER
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS question_intervals (
            user_id TEXT,
            question TEXT,
            last_answered_date TEXT,
            interval_days INTEGER,
            PRIMARY KEY (user_id, question)
        )
        ''')
        conn.commit()
        return conn, cursor
    
    # Function to check if a question is due for the user    
    def is_question_due(cursor, user_id, question, interval_days, survey_date):
        """
        Check if a question is due based on the entered survey date
        - Ensures questions are only asked again after the specified interval.
        - Prints the number of days remaining if the question is not due.
        """
        cursor.execute(
            '''
            SELECT last_answered_date FROM question_intervals WHERE user_id = ? AND question = ?
            ''', (user_id, question))
        result = cursor.fetchone()
        
        if result:
            last_date = datetime.strptime(result[0], "%Y-%m-%d")
            survey_date_obj = datetime.strptime(survey_date, "%Y-%m-%d")
            days_since_last = (survey_date_obj - last_date).days
            days_remaining = interval_days - days_since_last
            
            if days_remaining > 0:
                print(f"This question is not due yet. Please come back after {days_remaining} days.")
                return False
        return True
    
    # Function to update the last answered date for a question    
    def update_question_date(cursor, conn, user_id, question, interval_days, survey_date):
        """
        Update the last answered date for a question in the `question_intervals` table.
        """        
        cursor.execute('''
        INSERT OR REPLACE INTO question_intervals (user_id, question, last_answered_date, interval_days)
        VALUES (?, ?, ?, ?)
        ''', (user_id, question, survey_date, interval_days))
        conn.commit()
    
    # Function to export all survey responses    
    def export_all_responses():
        """Export all responses to a text file"""
        conn, cursor = setup_database()
        cursor.execute('''SELECT * FROM survey_responses''')
        rows = cursor.fetchall()
        filename = "all_user_data.txt"
        with open(filename, "w") as file:
            for row in rows:
                file.write(f"User ID: {row[1]}, Survey Date: {row[2]}, Question: {row[3]}, Response: {row[4]}\n")
        print(f"All responses exported to {filename}")
    
    # Main survey function    
    def run_survey():
        """
        Conduct the survey using a Turtle-based interface.

        - Presents questions to the user and collects responses.
        - Ensures questions respect their specified time intervals.
        - Saves responses to the database and exports them to a text file.
        """

        # Setting up the Turtle screen
        turtle.TurtleScreen._RUNNING = True
        screen = turtle.Screen()
        screen.clear()
        screen.setup(width=800, height=700)
        screen.title("Measuring Loneliness Survey")
        screen.bgcolor("white")
        
        # Connect to the database
        conn, cursor = setup_database()
    
        # Get the user's name and anonymize it
        user_name = screen.textinput("User Name", "Enter your name:")
        if not user_name:
            print("User name is required.")
            screen.bye()
            return
        user_id = anonymize_name(user_name)
    
        # Get survey date from the user
        while True:
            survey_date = screen.textinput("Survey Date", "Enter the survey date (YYYY-MM-DD):")
            try:
                survey_date_obj = datetime.strptime(survey_date, "%Y-%m-%d")
                if not (1 <= survey_date_obj.month <= 12 and 1 <= survey_date_obj.day <= 31):
                    raise ValueError
                survey_date = survey_date_obj.strftime("%Y-%m-%d")
                break
            except ValueError:
                print("Invalid date. Ensure the month is between 01 and 12, and the day is between 01 and 31.")
    
        # Initialize pen
        pen = turtle.Turtle()
        pen.hideturtle()
        pen.speed(0)
        
        # Questions and scales with intervals
        questions_with_scales = {
            "Weekly: How many hours did you spend alone last week?": {
                "interval": 7,
                "options": [
                    "0-20 hours",
                    "21-40 hours",
                    "41-80 hours",
                    "81-120 hours",
                    "121-168 hours"
                ]
            },
            "Weekly: How would you rate your physical health?": {
                "interval": 7,
                "options": [
                    "Poor",
                    "Fair",
                    "Good",
                    "Very good",
                    "Excellent"
                ]
            },
            "Weekly: How would you rate your mental health?": {
                "interval": 7,
                "options": [
                    "Poor",
                    "Fair",
                    "Good",
                    "Very good",
                    "Excellent"
                ]
            },
            "Quarterly: How often have you had face-to-face conversations in the past three months?": {
                "interval": 91,
                "options": [
                    "Not in the past three months",
                    "Less than monthly",
                    "Monthly",
                    "A few times a month",
                    "Weekly",
                    "A few times a week",
                    "Daily or almost daily"
                ]
            },
            "Quarterly: How often have you volunteered in the past three months?": {
                "interval": 91,
                "options": [
                    "Not in the past three months",
                    "Less than monthly",
                    "Monthly",
                    "A few times a month",
                    "Weekly",
                    "A few times a week",
                    "Daily or almost daily"
                ]
            }
        }
        
        # Process each question and collect responses        
        responses = {}
        any_questions_available = False
        for question, config in questions_with_scales.items():
            if not is_question_due(cursor, user_id, question, config["interval"], survey_date):
                continue
                
            any_questions_available = True
            pen.clear()
            pen.penup()
            pen.goto(0, 250)
            pen.write(question, align="center", font=("Arial", 16, "bold"))
    
            # Display options
            for i, label in enumerate(config["options"]):
                # Position the Turtle pen for writing the option text
                # Generated by ChatGPT: The y-coordinate decreases with each option to stack them vertically & this ensures the options are evenly spaced, with 30 pixels between each.               
                pen.goto(0, 200 - i * 30)
                # Write the option text on the screen using the specified font and alignment               
                pen.write(f"{i}: {label}", align="center", font=("Arial", 12, "normal"))
    
            # Get user input
            while True:
                # Prompt the user to input their answer using a Turtle text box      
                # The valid range of inputs (0 to the number of options - 1) is displayed in the prompt                
                user_input = screen.textinput("Your Answer", f"Select a number (0-{len(config['options'])-1}):")
                # Validate the user input:        
                # - Ensure it is not empty
                # - Check if it is a digit                
                # - Confirm the input is within the valid range of option indices                
                if user_input and user_input.isdigit() and 0 <= int(user_input) < len(config["options"]):
                    # Save the user's response for the current question                    
                    responses[question] = int(user_input)
                    # Update the last answered date for the question in the database
                    # This ensures the question will only be asked again after the specified interval                    
                    update_question_date(cursor, conn, user_id, question, config["interval"], survey_date)
                    break  # Exit the loop once a valid response is provided

        # Check if no questions were available for the user based on intervals
        if not any_questions_available:
            pen.clear()
            pen.write("No questions are currently due. Please check back later.", align="center", font=("Arial", 16, "bold"))
            # Wait for a user click to close the Turtle window and terminate the survey            
            screen.exitonclick()
            return

        # Save responses to database
        for question, response in responses.items():
            cursor.execute('''
            INSERT INTO survey_responses (user_id, survey_date, question, response)
            VALUES (?, ?, ?, ?)
            ''', (user_id, survey_date, question, response))
        # Commit the changes to ensure the data is stored in the database        
        conn.commit()
    
        # Export all responses
        export_all_responses()
    
        # Clear the screen and thank the user for completing the survey    
        pen.clear()
        pen.goto(0, 0)
        pen.write("Survey completed. Thank you!", align="center", font=("Arial", 16, "bold"))
        
        # Wait for a user click to close the Turtle window and terminate the program        
        screen.exitonclick()
        screen.bye()
    
    # Prompt the user to confirm whether they want to delete all existing survey data.
    response = input("Do you want to delete all existing survey data and start fresh? (yes/no): ")
    if response.lower() == 'yes':
        reset_survey_data()
    
    # Run the survey with error handling
    try:
        # Start the survey and handle any runtime errors        
        run_survey()
    except turtle.Terminator:
        # Handle the case where the Turtle window is closed prematurely by the user        
        print("Survey window was closed.")
    except Exception as e:
        # Print any unexpected errors that occur during the survey process        
        print(f"An error occurred: {str(e)}")
   

# Module 6: Calculate and save user statistics
def calculate_user_statistics(file_path, output_file="user_statistics.txt"):
    """
    Calculate mean and median from user data and save the statistics to a file.

    Parameters:
        file_path (str): Path to the input user data file containing survey responses.
        output_file (str): Path to save the calculated statistics. Defaults to "user_statistics.txt".

    Output:
        - A text file containing mean and median statistics for each survey category.
    """

    # Initialize dictionaries to store responses for each category
    values = {
        "Hours Alone": [],
        "Physical Health": [],
        "Mental Health": [],
        "Face-to-Face Conversations": [],
        "Volunteering": []
    }
        
    # Read and parse user data from the input file
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the current line corresponds to a specific question category  
            # Extract the response value and add it to the appropriate category list            
            if "How many hours did you spend alone" in line:
                values["Hours Alone"].append(int(line.split("Response: ")[1]))
            elif "physical health" in line.lower():
                values["Physical Health"].append(int(line.split("Response: ")[1]))
            elif "mental health" in line.lower():
                values["Mental Health"].append(int(line.split("Response: ")[1]))
            elif "face-to-face conversations" in line.lower():
                values["Face-to-Face Conversations"].append(int(line.split("Response: ")[1]))
            elif "volunteered" in line.lower():
                values["Volunteering"].append(int(line.split("Response: ")[1]))
        
    # Calculate statistics (mean and median) and write them to the output file
    with open(output_file, 'w') as out_file:
        # Write a header to the output file        
        out_file.write("User Statistics:\n\n")
        # Iterate through each category in the dictionary        
        for category, data in values.items():
            if data:
                # Calculate the mean of the collected responses                
                mean = sum(data) / len(data)
                # Calculate the median of the collected responses                
                sorted_data = sorted(data)
                mid = len(sorted_data) // 2
                median = sorted_data[mid] if len(sorted_data) % 2 != 0 else (sorted_data[mid-1] + sorted_data[mid]) / 2
                
                # Add "hours" unit for the "Hours Alone" category                
                unit = " hours" if category == "Hours Alone" else ""
                
                # Write the statistics to the output file                
                out_file.write(f"{category}:\n")
                out_file.write(f"  Mean: {mean:.2f}{unit}\n")
                out_file.write(f"  Median: {median:.2f}{unit}\n\n")

    # Print a confirmation message    
    print(f"User statistics have been saved to {output_file}")



# Module 7: Parsing and Comparing Statistics
def compare_statistics(user_stats_file, benchmark_file):
    """
    Compare user statistics with benchmark values and generate detailed analysis
    
    Parameters:
        user_stats_file (str): Path to the file containing user statistics
        benchmark_file (str): Path to the file containing benchmark statistics

    Returns:
        list: A list of dictionaries, each containing a comparison result
    """

    def parse_stats_file(filepath):
        """
        Parse a statistics file and extract mean and median values.

        Parameters:
            filepath (str): Path to the statistics file.

        Returns:
            dict: A dictionary with metrics as keys and their mean/median values as subkeys.
        """
        
        stats = {}
        current_metric = None # Keeps track of the current metric being parsed
        
        # Open the file and parse each line        
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip() # Remove leading/trailing whitespace
                
                # Skip empty lines and headers
                if not line or line.startswith(("User Statistics:", "Benchmarks for", "=")):
                    continue
                    
                # Check for metric lines (ends with colon)
                if line.endswith(':'):
                    current_metric = line[:-1]  # Remove trailing colon
                    stats[current_metric] = {}
                    
                # Parse mean and median values under the current metric
                elif current_metric and ('Mean:' in line or 'Median:' in line):
                    # Split on colon and handle potential "hours" suffix
                    stat_type, value_part = [x.strip() for x in line.split(':')]
                    stat_type = stat_type.lower()  # Convert to lowercase for consistent comparison
                    
                    # Extract numeric value, removing any units
                    value_str = value_part.split()[0]  # Take first part before any units
                    try:
                        value = float(value_str)
                        stats[current_metric][stat_type] = value
                    except ValueError as e:
                        print(f"Warning: Could not parse value in line: {line} - {e}")

        # Debug print for individual file parsing
        print(f"\nParsing {filepath}:")
        print("Found metrics:", list(stats.keys()))
        for metric, values in stats.items():
            print(f"{metric}: {values}")
            
        return stats

    # Mapping between user stats and benchmark metrics
    metric_mapping = {
        'Hours Alone': 'CONNECTION_social_time_alone',
        'Physical Health': 'WELLNESS_self_rated_physical_health',
        'Mental Health': 'WELLNESS_self_rated_mental_health',
        'Face-to-Face Conversations': 'CONNECTION_activities_face_to_face_convorsation_p3m',
        'Volunteering': 'CONNECTION_activities_volunteered_p3m'
    }

    try:
        # Parse both user and benchmark statistics files
        user_stats = parse_stats_file(user_stats_file)
        benchmark_stats = parse_stats_file(benchmark_file)
        
        if not user_stats or not benchmark_stats:
            print("Error: Failed to parse one or both statistics files.")
            return []
        
        results = []
        
        # Compare user statistics against benchmarks        
        for user_metric, bench_metric in metric_mapping.items():
            if user_metric in user_stats and bench_metric in benchmark_stats:
                for stat_type in ['mean', 'median']:
                    if stat_type in user_stats[user_metric] and stat_type in benchmark_stats[bench_metric]:
                        user_val = user_stats[user_metric][stat_type]
                        bench_val = benchmark_stats[bench_metric][stat_type]
                        
                        # Calculate absolute and percentage differences
                        abs_diff = user_val - bench_val
                        pct_diff = (abs_diff / bench_val * 100) if bench_val != 0 else float('inf')
                        
                        # Determine trend 
                        if abs(pct_diff) <= 5:
                            trend = "Stable"
                        else:
                            trend = "Higher" if pct_diff > 0 else "Lower"
                        
                        # Assess significance
                        if abs(pct_diff) <= 10:
                            significance = "Normal Range"
                        elif abs(pct_diff) <= 20:
                            significance = "Moderate"
                        else:
                            significance = "Significant"
                        
                        # Append comparison result                        
                        results.append({
                            'metric': user_metric,
                            'stat_type': stat_type.capitalize(),
                            'user_value': user_val,
                            'benchmark_value': bench_val,
                            'absolute_difference': abs_diff,
                            'percentage_difference': pct_diff,
                            'trend': trend,
                            'significance': significance
                        })
        
        return results
    
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# Module 8: Generating the comparison_report
def generate_comparison_report(results, output_file="comparison_report.txt"):
    """
    Generate a detailed comparison report from the analysis results.
    
    Parameters:
        results (list): A list of dictionaries containing comparison results.
                        Each dictionary includes:
                            - metric: Name of the metric being compared.
                            - stat_type: Type of statistic (e.g., Mean, Median).
                            - user_value: User's calculated value for the metric.
                            - benchmark_value: Benchmark value for the metric.
                            - absolute_difference: Difference between user and benchmark values.
                            - percentage_difference: Percent difference from the benchmark.
                            - trend: Qualitative trend (e.g., Stable, Higher, Lower).
                            - significance: Assessment of the difference (e.g., Normal Range, Moderate, Significant).
        output_file (str): Path where the generated report will be saved. Defaults to "comparison_report.txt".

    """
    try:
        # Open the specified output file in write mode        
        with open(output_file, 'w') as f:
            # Write a header for the report            
            f.write("Senior Well-Being Analysis - Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            current_metric = None  # Track the current metric to avoid redundant headers
            
            # Iterate through the comparison results to generate the report            
            for result in results:
                # Print header for new metrics
                if current_metric != result['metric']:
                    current_metric = result['metric']
                    f.write(f"\n{current_metric}\n")   # Write the metric name as a header
                    f.write("-" * len(current_metric) + "\n")  # Underline the metric header
                
                f.write(f"\n{result['stat_type']} Analysis:\n")  # Specify Mean or Median
                f.write(f"  Your Value:          {result['user_value']:.2f}\n")  # User's calculated value 
                f.write(f"  Benchmark:           {result['benchmark_value']:.2f}\n")  # Benchmark value
                f.write(f"  Absolute Difference: {result['absolute_difference']:+.2f}\n")  # Absolute difference
                f.write(f"  Percentage Change:   {result['percentage_difference']:+.1f}%\n")  # Percent difference
                f.write(f"  Trend:               {result['trend']}\n")  # Qualitative trend
                f.write(f"  Assessment:          {result['significance']}\n")  # Assessment of the difference
                
        # Confirm successful report generation                
        print(f"\nDetailed comparison report has been saved to {output_file}")
        
    except Exception as e:
        # Handle and report any errors during the report generation process        
        print(f"Error generating report: {str(e)}")

def handle_comparison_and_report():
    """
        Handle the process of comparing statistics and generating a detailed report.
    
        This function:
            - Prompts the user to provide paths to user statistics and benchmark files.
            - Validates the provided file paths.
            - Generates comparison results by calling `compare_statistics`.
            - Displays a summary of comparison results in the console.
            - Writes a detailed comparison report to a file.
            - Optionally allows the user to view the generated report in the console.
    """

    print("\nCompare Statistics and Generate Report")
    print("=" * 40)
    
    # Get file paths with input validation
    while True:
        # Get the user statistics file path or use the default        
        user_stats_file = input("\nEnter path to user statistics file (default: user_statistics.txt): ").strip()
        if not user_stats_file:
            user_stats_file = "user_statistics.txt"
        if not user_stats_file.endswith('.txt'):
            user_stats_file += '.txt'
            
        # Get the benchmark file path or use the default            
        benchmark_file = input("Enter path to benchmarks file (default: benchmarks.txt): ").strip()
        if not benchmark_file:
            benchmark_file = "benchmarks.txt"
        if not benchmark_file.endswith('.txt'):
            benchmark_file += '.txt'
            
        # Check if both files exist
        if not os.path.exists(user_stats_file):
            print(f"Error: User statistics file '{user_stats_file}' not found.")
            continue
        if not os.path.exists(benchmark_file):
            print(f"Error: Benchmark file '{benchmark_file}' not found.")
            continue
        break  # Exit the loop if both files exist
    
    try:
        # Generate comparison results
        print(f"\nComparing statistics from:")
        print(f"User file: {user_stats_file}")
        print(f"Benchmark file: {benchmark_file}\n")
        
        # Perform the comparison using the provided files        
        results = compare_statistics(user_stats_file, benchmark_file)
        
        if results:
            # Print a summary of the comparison results in a tabular format
            print("Comparison Results:")
            print("=" * 100)
            print(f"{'Metric':<30} {'Type':<8} {'User':<10} {'Benchmark':<10} {'% Diff':<10} {'Trend':<10} {'Assessment'}")
            print("-" * 100)
            
            # Print comparison results
            for result in results:
                print(f"{result['metric']:<30} "
                      f"{result['stat_type']:<8} "
                      f"{result['user_value']:<10.2f} "
                      f"{result['benchmark_value']:<10.2f} "
                      f"{result['percentage_difference']:>+9.1f}% "
                      f"{result['trend']:<10} "
                      f"{result['significance']}")
            
            # Generate a detailed report and save it to a file
            output_file = "comparison_report.txt"
            generate_comparison_report(results, output_file)
            
            # Ask if user wants to view the report
            view_report = input("\nWould you like to view the detailed report now? (y/n): ").lower()
            if view_report == 'y':
                try:
                    # Read and display the report file                    
                    with open(output_file, 'r') as f:
                        print(f"\n{f.read()}")
                except FileNotFoundError:
                    print("Error: Report file not found.")
                
        else:
            print("No matching metrics found for comparison. Please check your input files.")
            
    except Exception as e:
        # Handle any errors during comparison or report generation        
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your input files and try again.")


# Main Menu
def main_menu():
    """
    Main menu for interacting with the program    
    """
    print("Welcome To The Senior Well-Being Analysis Tool!")
    file_path = input("Enter the path to your dataset: ")
    try:
        cleaned_data = load_and_clean_data(file_path)
        print("\nData loaded and cleaned successfully!")
        print(f"Dataset contains {cleaned_data.shape[0]} rows and {cleaned_data.shape[1]} columns.")
        
        # Ensure columns are numeric
        columns_to_numeric = [
            "LONELY_dejong_emotional_social_loneliness_scale_TOTAL",
            "CONNECTION_activities_face_to_face_convorsation_p3m",
            "WELLNESS_self_rated_physical_health",
            "WELLNESS_self_rated_mental_health",
            "CONNECTION_social_time_alone",
            "CONNECTION_activities_volunteered_p3m"
        ]

        for column in columns_to_numeric:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric, coerce errors to NaN
            else:
                print(f"Warning: Column {column} not found in the dataset.")

        # Drop rows with missing values in these columns
        df_cleaned = df.dropna(subset=columns_to_numeric)

        #Menu Loop        
        while True:
            print("\nMain Menu:")
            print("1. View Cleaned Data") 
            print("2. Generate Visualizations")
            print("3. View Benchmark Values")
            print("4. Enter New Data")
            print("5. Calculate User Statistics")       
            print("6. Compare Statistics with Benchmark Values and Generate Report")  # Combined option
            print("7. Exit")

            choice = input("Select an option (1-6): ")

            if choice == '1':
                print("\nCleaned Data:")
                print(cleaned_data.to_string(index=False))
            elif choice == '2':
                generate_visualizations(cleaned_data)
            elif choice == '3':
                print("Viewing benchmark values...")
                save_benchmarks_to_file(benchmarks)
            elif choice == '4':
                # Call Turtle-based data entry function
                print("\nEntering new data...")
                turtle_based_data_entry()  
            elif choice == '5':
                user_data_file = input("Enter path to your user data file: ")
                calculate_user_statistics(user_data_file)
            elif choice == '6':
                handle_comparison_and_report()           
            elif choice == '7':   
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid choice. Please select a valid option.")
    except FileNotFoundError:
        print("Error: File not found. Please check the file path and try again.")

if __name__ == "__main__":       
    main_menu()
