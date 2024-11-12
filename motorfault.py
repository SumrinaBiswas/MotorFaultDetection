

import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.fft import fft, fftfreq
import random
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

## Loading all sheets from both excel files in a dictionary

ideals=pd.read_excel('ideal.xlsx', sheet_name=None)
nonIdeals=pd.read_excel('nonideal.xlsx',sheet_name=None)

print(ideals.keys())
print(nonIdeals.keys())

ideal_sheets = {index: key for index, key in enumerate(ideals.keys())}
nonIdeal_sheets = {index: key for index, key in enumerate(nonIdeals.keys())}
print(f'\n========================================================================================')
print(ideal_sheets)
print(nonIdeal_sheets)

#Function to plot motor behavior under a particular scenario as index, motor_type(Ideal or non-Ideal) and phase(int) as mentioned by user.

def plot_motor_behavior(motor_type, indices, phase):
    """
    Plots the behavior of motors based on user inputs for motor type, conditions, and phase.

    Parameters:
    motor_type (int): 1 for IDEAL, 2 for NON-IDEAL, 3 for BOTH.
    indices (DataFrame): DataFrame containing 'ideal_conditions' and 'nonIdeal_conditions'.
    phase (int): Phase to compare (1, 2, or 3).

    Returns:
    None: Displays the plot or returns an error message if invalid input.
    """

    # Function to plot data for given conditions
    def plot_conditions(conditions, motor_type_label, motor_sheets, motor_data, phase):
        for index in conditions:
            sheet_name = motor_sheets[index]
            df = motor_data[sheet_name]
            color = np.random.rand(3,)  # Random color for each plot line
            plt.plot(df['Relative Time'], df.iloc[:, phase], label=f"{motor_type_label} Motor - {sheet_name}", color=color)

    # Start plotting based on motor type
    plt.figure(figsize=(15, 7.5))

    if motor_type == 1:  # IDEAL motor
        conditions = indices['ideal_conditions'].dropna().tolist()
        plot_conditions(conditions, 'Ideal', ideal_sheets, ideals, phase)
        plt.title(f"Phase {phase} Comparison for Ideal Motor")

    elif motor_type == 2:  # NON-IDEAL motor
        conditions = indices['nonIdeal_conditions'].dropna().tolist()
        plot_conditions(conditions, 'Non-Ideal', nonIdeal_sheets, nonIdeals, phase)
        plt.title(f"Phase {phase} Comparison for Non-Ideal Motor")

    elif motor_type == 3:  # BOTH motors
        ideal_conditions = indices['ideal_conditions'].dropna().tolist()
        non_ideal_conditions = indices['nonIdeal_conditions'].dropna().tolist()

        if len(ideal_conditions) > 0:
            plot_conditions(ideal_conditions, 'Ideal', ideal_sheets, ideals, phase)
        if len(non_ideal_conditions) > 0:
            plot_conditions(non_ideal_conditions, 'Non-Ideal', nonIdeal_sheets, nonIdeals, phase)

        plt.title(f"Phase {phase} Comparison for Ideal and Non-Ideal Motors")

    plt.xlabel('Relative Time')
    plt.ylabel(f'Phase {phase} Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def user_input():
    print("Choose motor type:")
    print("1. IDEAL")
    print("2. NON-IDEAL")
    print("3. BOTH (Ideal vs Non-Ideal)")

    # Validating motor_type input
    while True:
        motor_type = input("Enter the number corresponding to the motor type: ")
        if motor_type in ['1', '2', '3']:
            motor_type = int(motor_type)
            break
        else:
            print("Invalid motor type. Please choose '1' for IDEAL, '2' for NON-IDEAL, or '3' for BOTH.")

    # Displaying relevant sheets based on motor type
    if motor_type == 1:
        print("IDEAL Motor Sheets:")
        print(ideal_sheets)
    elif motor_type == 2:
        print("NON-IDEAL Motor Sheets:")
        print(nonIdeal_sheets)
    elif motor_type == 3:
        print("IDEAL Motor Sheets:")
        print(ideal_sheets)
        print("NON-IDEAL Motor Sheets:")
        print(nonIdeal_sheets)

    # Asking for the number of conditions to plot
    total_conditions = int(input("Enter the total number of conditions you want to plot: "))

    # Initialize lists to store ideal and non-ideal conditions
    ideal_conditions = []
    nonIdeal_conditions = []

    for i in range(total_conditions):
        # Asking for conditions, allowing users to switch between Ideal and Non-Ideal
        condition_type = input(f"Enter 'i' for IDEAL or 'n' for NON-IDEAL for condition {i+1}: ").lower()

        if condition_type == 'i':
            index = int(input(f"Enter the IDEAL sheet index {i+1}: "))
            ideal_conditions.append(index)
        elif condition_type == 'n':
            index = int(input(f"Enter the NON-IDEAL sheet index {i+1}: "))
            nonIdeal_conditions.append(index)
        else:
            print("Invalid choice. Please enter 'i' for IDEAL or 'n' for NON-IDEAL.")
            i -= 1  # Retry this iteration

    # Construct DataFrame using pd.Series to handle unequal lengths
    conditions_df = pd.DataFrame({
        'ideal_conditions': pd.Series(ideal_conditions),
        'nonIdeal_conditions': pd.Series(nonIdeal_conditions)
    })

    # Input for phase and validation
    while True:
        phase = input("Enter the phase (1, 2, or 3): ")
        if phase in ['1', '2', '3']:
            phase = int(phase)
            break
        else:
            print("Invalid phase. Please choose 1, 2, or 3.")

    return motor_type, conditions_df, phase


def main():
    motor_type, conditions_df, phase = user_input()
    plot_motor_behavior(motor_type, conditions_df, phase)

main()

# Decomposing TIME-DOMAIN Data to FREQUENCY-DOMAIN Data

def get_user_input():
    """
    Get the necessary input parameters from the user, except for sampling rate which will be calculated dynamically.
    """
    # Asking the user to choose between Ideal and Non-Ideal motor data
    motor_type = int(input("Select motor type: \n1. Ideal\n2. Non-Ideal\nEnter choice: "))

    # Validate motor_type input
    if motor_type not in [1, 2]:
        print("Invalid choice. Please choose either 1 or 2.")
        return get_user_input()

    # Based on motor type, either display Ideal or Non-Ideal sheets
    if motor_type == 1:
        sheets = list(ideals.keys())
        print("Ideal Motor Sheets:")
    else:
        sheets = list(nonIdeals.keys())
        print("Non-Ideal Motor Sheets:")

    # Listing available sheets and letting the user select a sheet
    for idx, sheet in enumerate(sheets):
        print(f"{idx + 1}. {sheet}")

    sheet_number = int(input("Enter the number corresponding to the sheet you want to select: ")) - 1
    selected_sheet = sheets[sheet_number]

    # Asking the user to select the phase as an integer
    phase = int(input("Enter the phase to analyze (1 for U, 2 for V, or 3 for W): "))
    if phase not in [1, 2, 3]:
        print("Invalid phase. Please choose 1, 2, or 3.")
        return get_user_input()

    # Returning selected parameters (without sampling rate)
    return motor_type, selected_sheet, phase

def calculate_sampling_rate(time_column):
    """
    Calculate the sampling rate from the time data.
    :param time_column: The column containing time values (Relative Time).
    :return: Sampling rate in Hz.
    """
    time_diffs = np.diff(time_column)  # Calculate differences between consecutive time points
    avg_time_step = np.mean(time_diffs)  # Average time difference
    sampling_rate = 1.0 / avg_time_step  # Sampling rate is the inverse of the average time difference
    return sampling_rate

def decompose_to_frequency(df, phase_column, sampling_rate):
    """
    Decompose the time-domain data from the selected phase to the frequency domain using FFT.
    :param df: DataFrame containing time-domain data (from the selected sheet).
    :param phase_column: The selected phase column (U, V, or W) from the DataFrame.
    :param sampling_rate: Sampling rate in Hz (calculated dynamically from the data).
    """
    N = len(phase_column)  # Length of the signal
    phase_column_np = phase_column.to_numpy().astype(np.float64)
    yf = fft(phase_column_np)  # FFT on the time-domain signal
    xf = fftfreq(N, 1 / sampling_rate)  # Frequency bins

    # Considering only positive frequencies
    xf = xf[:N // 2]
    yf = 2.0 / N * np.abs(yf[:N // 2])  # Magnitude of FFT

    return xf, yf

def plot_frequency_spectrum(xf, yf, phase):
    """
    Plot the frequency spectrum of the selected phase.
    :param xf: Frequency values (FFT frequencies).
    :param yf: Magnitudes of the FFT.
    :param phase: Motor phase (1, 2, or 3).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(xf, yf)
    plt.title(f"Frequency Spectrum for Phase {phase}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

def identify_key_frequencies(xf, yf, num_freqs=5):
    """
    Identify the most significant frequencies in the spectrum.
    :param xf: Frequency values (FFT frequencies).
    :param yf: Magnitudes of the FFT.
    :param num_freqs: Number of key frequencies to identify.
    :return: List of key frequencies and their corresponding magnitudes.
    """
    # Sorting frequencies by magnitude in descending order
    key_freq_indices = np.argsort(yf)[-num_freqs:][::-1]
    key_frequencies = xf[key_freq_indices]
    key_magnitudes = yf[key_freq_indices]

    print(f"Top {num_freqs} key frequencies and magnitudes:")
    for freq, mag in zip(key_frequencies, key_magnitudes):
        print(f"Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")

    return key_frequencies, key_magnitudes

def main():
    # Get user input for parameters (motor type, sheet, and phase as integer)
    motor_type, selected_sheet, phase = get_user_input()

    # Load the corresponding DataFrame based on motor type
    if motor_type == 1:
        df = ideals[selected_sheet]
    else:
        df = nonIdeals[selected_sheet]

    # Calculate the sampling rate from the 'Relative Time' column
    time_column = df['Relative Time']
    sampling_rate = calculate_sampling_rate(time_column)

    # Select the phase column based on user input
    phase_column = df.iloc[:, phase]

    # Decompose the selected phase's time-domain data to frequency domain
    xf, yf = decompose_to_frequency(df, phase_column, sampling_rate)

    # Plot the frequency spectrum
    plot_frequency_spectrum(xf, yf, phase)

    # Identify the top 5 key frequencies
    identify_key_frequencies(xf, yf)

# Execute the main function
main()

# PLOTTING HARMONICS(PEAKS MARKED).

# Function to calculate sampling rate dynamically based on time intervals
def calculate_sampling_rate(df):
    time_column = df['Relative Time'].to_numpy()
    time_diffs = np.diff(time_column)  # Calculate differences between consecutive time points
    avg_time_diff = np.mean(time_diffs)  # Get the average time difference
    sampling_rate = 1 / avg_time_diff  # Sampling rate is the reciprocal of the time difference
    return sampling_rate

def decompose_to_frequency(df, phase_column, sampling_rate):
    """
    Perform FFT on the signal and return the positive frequency components and corresponding magnitudes.
    """
    # Extract the signal for the given phase
    signal = df.iloc[:, phase_column].values  # Access the values using .values

    # Convert signal to a contiguous array with the correct data type
    signal = np.ascontiguousarray(signal, dtype=np.float64)

    N = len(signal)  # Length of the signal
    yf = fft(signal)  # FFT on the time-domain signal
    xf = fftfreq(N, 1 / sampling_rate)  # Frequency bins

    # Only take the positive frequencies and magnitudes
    harmonic_data = pd.DataFrame({
        'Frequency (Hz)': xf[:N // 2],  # Positive frequencies
        'Magnitude': 2.0 / N * np.abs(yf[:N // 2])  # Positive magnitudes
    })

    # Find the fundamental frequency (the frequency with the maximum magnitude)
    fundamental_frequency_idx = harmonic_data['Magnitude'].idxmax()
    fundamental_frequency = harmonic_data['Frequency (Hz)'][fundamental_frequency_idx]

    # Create harmonic levels based on the fundamental frequency
    harmonic_levels = [fundamental_frequency * (i + 1) for i in range(19)]  # 1st to 19th harmonics
    harmonic_data['Harmonic'] = np.round(harmonic_data['Frequency (Hz)'] / fundamental_frequency)

    return harmonic_data, fundamental_frequency_idx, harmonic_levels

# Function to calculate harmonics and return data for plotting
def calculate_harmonics_for_plotting(ideals, nonIdeals, phase):
    results = []

    for motor_name, motor_type in [("Ideal", ideals), ("Non-Ideal", nonIdeals)]:
        for sheet_name, df in motor_type.items():
            # Calculate the sampling rate dynamically
            sampling_rate = calculate_sampling_rate(df)

            # Decompose to frequency
            harmonic_data, fundamental_frequency_idx, harmonic_levels = decompose_to_frequency(df, phase, sampling_rate)

            # Store results
            results.append({
                'Motor Type': motor_name,
                'Sheet Name': sheet_name,
                'Harmonic Data': harmonic_data,
                'Fundamental Frequency Index': fundamental_frequency_idx,
                'Harmonic Levels': harmonic_levels
            })

    return results

def plot_harmonics(harmonic_data, fundamental_frequency_idx):
    plt.figure(figsize=(10, 6))

    # Frequency and magnitude data
    freq = harmonic_data['Frequency (Hz)']
    mag = harmonic_data['Magnitude']

    # Filter out the fundamental frequency
    fundamental_frequency = freq[fundamental_frequency_idx]
    harmonic_data_filtered = harmonic_data.drop(fundamental_frequency_idx)

    # Plot the filtered harmonic data
    plt.plot(harmonic_data_filtered['Frequency (Hz)'], harmonic_data_filtered['Magnitude'], label='Harmonics', color='blue')

    # Mark all harmonic levels
    for level in range(1, 20):
        harmonic_freq = fundamental_frequency * level
        if harmonic_freq in harmonic_data_filtered['Frequency (Hz)'].values:
            harmonic_amplitude = harmonic_data_filtered.loc[harmonic_data_filtered['Frequency (Hz)'] == harmonic_freq, 'Magnitude'].values[0]
            plt.plot(harmonic_freq, harmonic_amplitude, 'ro')  # Mark harmonics in red
            plt.text(harmonic_freq, harmonic_amplitude, f'H{level}', fontsize=9, color='red', ha='center')

    # Plot labels and title
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency vs Amplitude of Harmonics (Excluding Fundamental Frequency)')
    plt.axvline(x=fundamental_frequency, color='gray', linestyle='--', label='Fundamental Frequency')  # Optional line for fundamental frequency
    plt.legend()
    plt.grid(True)
    plt.show()

# Example to use the plot function with the results obtained
def main():
    phase = int(input(f"\nEnter the phase (1, 2, or 3) for harmonics: "))

    # Calculate harmonics for plotting
    results = calculate_harmonics_for_plotting(ideals, nonIdeals, phase)

    # Plot each motor's harmonic data
    for result in results:
        motor_name = result['Motor Type']
        sheet_name = result['Sheet Name']
        harmonic_data = result['Harmonic Data']
        fundamental_frequency_idx = result['Fundamental Frequency Index']

        print(f"Plotting for {motor_name} Motor - {sheet_name}...")
        plot_harmonics(harmonic_data, fundamental_frequency_idx)

# Run the main function
main()

def decompose_to_frequency(df, phase_column, sampling_rate):
    """
    Perform FFT on the signal and return the positive frequency components and corresponding magnitudes.
    """
    signal = df.iloc[:, phase_column].values
    signal = np.ascontiguousarray(signal, dtype=np.float64)

    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)

    # Only take the positive frequencies and magnitudes
    harmonic_data = pd.DataFrame({
        'Frequency (Hz)': xf[:N // 2],
        'Magnitude': 2.0 / N * np.abs(yf[:N // 2])
    })

    return harmonic_data


#Creating 2 excel file(for ideal and non-ideal motor) with multiple sheets which will include all harmonics level and data(frequency and amplitude).

# Function to create harmonic DataFrame for Ideal or Non-Ideal motor data
def create_harmonic_dataframe(motor_data, file_name):
    with pd.ExcelWriter(file_name, mode='w') as writer:
        for phase in range(1, 4):  # Phases 1, 2, 3
            harmonics_data = []
            for condition_name, df in motor_data.items():
                sampling_rate = calculate_sampling_rate(df)
                harmonic_data = decompose_to_frequency(df, phase, sampling_rate)
                fundamental_freq = harmonic_data.loc[harmonic_data['Magnitude'].idxmax(), 'Frequency (Hz)']

                # Gather harmonic amplitudes for 1st to 19th harmonics
                harmonics_dict = {'Condition Name': condition_name}
                harmonics_dict.update({f'H{level} Amplitude': harmonic_data.loc[
                    (np.abs(harmonic_data['Frequency (Hz)'] - fundamental_freq * level)).idxmin(), 'Magnitude']
                    for level in range(1, 20)})
                harmonics_data.append(harmonics_dict)

            # Create DataFrame, and write to sheet
            harmonics_df = pd.DataFrame(harmonics_data)
            harmonics_df.to_excel(writer, sheet_name=f'Phase {phase}', index=False)


def create_harmonics_excel(ideals, nonIdeals):
    create_harmonic_dataframe(ideals, 'ideal_harmonics_data.xlsx')
    create_harmonic_dataframe(nonIdeals, 'nonideal_harmonics_data.xlsx')

create_harmonics_excel(ideals, nonIdeals)

# DATA Representation ===============================================================================================================
''' Data Preparation

First, we'll ensure that the data (harmonic amplitudes) is properly organized. We'll load the dataset of harmonics and prepare it for training a classification model. '''

ideal_file_path = 'ideal_harmonics_data.xlsx'
non_ideal_file_path = 'nonideal_harmonics_data.xlsx'

ideal_sheets = pd.read_excel('ideal_harmonics_data.xlsx', sheet_name=None)
non_ideal_sheets = pd.read_excel('nonideal_harmonics_data.xlsx', sheet_name=None)


ideal_harmonics_df = pd.concat(ideal_sheets.values(), ignore_index=True)
non_ideal_harmonics_df = pd.concat(non_ideal_sheets.values(), ignore_index=True)

ideal_harmonics_df['Target'] = 1
non_ideal_harmonics_df['Target'] = 0

combined_df = pd.concat([ideal_harmonics_df, non_ideal_harmonics_df], ignore_index=True)

X = combined_df.drop(columns=['Target'])
Y = combined_df['Target']

X.info()

numeric_X = X.select_dtypes(include=[float, int])
corr_matrix = numeric_X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Harmonic Features")
plt.show()

pca = PCA(n_components=5)
X_pca = pca.fit_transform(numeric_X)

plt.bar(range(1, 6), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Explained Variance by Principal Components')
plt.show()

loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(5)], index=numeric_X.columns)

# Display the top features contributing to each principal component
top_features_per_component = {}
for i in range(5):
    top_features = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5).index.tolist()
    top_features_per_component[f'PC{i+1}'] = top_features

print("Top 5 Features for each Principal Component:")
for component, features in top_features_per_component.items():
    print(f"{component}: {features}")


# Function to predict the main class using PCA and Random Forest
def predict_main_class(X, Y, n_pca_components=5, test_size=0.2, random_state=42):
    """
    This function performs PCA on the input features, selects important features using Random Forest,
    and predicts the main class.

    Parameters:
    - X: The feature set (excluding the target 'Condition Name' column).
    - Y: The target labels (main class).
    - n_pca_components: Number of PCA components to keep.
    - test_size: Proportion of the data to be used as the test set.
    - random_state: Random seed for reproducibility.

    Returns:
    - accuracy: Accuracy of the model on the test set.
    - rf_model: The trained Random Forest model.
    """

    # Step 1: Preprocess the data (scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X_scaled)

    # Step 3: Train a Random Forest on the PCA-reduced data
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_pca, Y)

    # Step 4: Determine important features based on Random Forest
    importances = rf.feature_importances_
    threshold = np.mean(importances)  # Use the mean importance as a threshold
    important_indices = np.where(importances >= threshold)[0]

    # Select the important PCA components
    X_important_pca = X_pca[:, important_indices]

    # Step 5: Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_important_pca, Y, test_size=test_size, random_state=random_state)

    # Step 6: Train a new Random Forest classifier on the selected important features
    rf_important = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_important.fit(X_train, Y_train)

    # Step 7: Make predictions on the test set
    Y_pred_important = rf_important.predict(X_test)

    # Step 8: Calculate the accuracy
    accuracy = accuracy_score(Y_test, Y_pred_important)
    print(f"Accuracy after PCA and Random Forest Feature Selection: {accuracy}")

    return accuracy, rf_important

# Example usage:
# Assuming X_numeric and Y are defined (X without 'Condition Name', Y with main class labels)
X_numeric = X.drop(columns=['Condition Name'])
accuracy, rf_model = predict_main_class(X_numeric, Y)

# ===================================== Subclass Prediction =================================


pca = PCA(n_components=7)
X_pca = pca.fit_transform(X_numeric)

clf_cross_val_score=cross_val_score(rf_model, X_pca, Y, cv=5)
clf_cross_val_score.mean(), clf_cross_val_score
print(f"\n\n")
print(clf_cross_val_score)

# Subclass Model

X_numeric = X.drop(columns=['Condition Name'])  # Features
Y_subclass = X['Condition Name']        # Target (subclass labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)  # Scale the features


#considering various type 1 & 2 error as Type-1 and Type-2 error
Y_subclass = Y_subclass.replace({'Type-1 variation B': 'Type-1 error',
                                 'Type-1 variation A': 'Type-1 error',
                                 'Type-1 variation C': 'Type-1 error',
                                 'Type-1 variation D': 'Type-1 error',
                                 'Type-2 variation A': 'Type-2 error',
                                 'Type-2 variation D': 'Type-2 error',
                                 'Type-2 variation E': 'Type-2 error',
                                 'Core fault-1': 'Core fault',
                                 'Core fault-2': 'Core fault'
                                 })

# Step 3: Apply PCA to reduce dimensionality (let's assume 7 components)
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, Y_subclass, test_size=0.15, random_state=42)

# ========= SVM Model ======

# Step 5: Train SVM model with linear kernel
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = svm_model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)


# Classification report
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy of SVM model: {accuracy}")

#======== Random Forest=============


X_train, X_test, y_train, y_test = train_test_split(X_pca, Y_subclass, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# Classification report
print("\nClassification Report (RandomForest):")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy of random forest classifier model: {accuracy}")

# ========== DecisionTree =============

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Classification report
print("\nClassification Report (DecisionTree):")
print(classification_report(y_test, y_pred))

print(f"\nAccuracy of Decision Tree Classifier model: {accuracy}")
