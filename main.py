import pandas as pd

data = pd.read_csv("PlayTennis.csv")

def compute_likelihood_probabilities(data):
    
    likelihood_tables = {}

    # Get unique class labels (Yes / No)
    classes = data['Play Tennis'].unique()

    # Iterate over each feature in the dataset 
    for feature in data.columns[:-1]:
        
        likelihood_tables[feature] = {} # make probability dictionary for each feature


        for label in classes:
            # Calculate conditional probabilities P(feature|label)
            probabilities = {}
            subset = data[data['Play Tennis'] == label][feature] #selects the subset of data where the 'Play Tennis' column matches the current class label ('Yes' or 'No')
            total_count = len(subset)
            unique_values = subset.unique()

            # Count occurrences of each value of the feature for example count how many sunny and yes 
            for value in unique_values:
                count = len(subset[subset == value])
                probabilities[value] = count / total_count

            likelihood_tables[feature][label] = probabilities

    # # Print likelihood tables
    # for feature, table in likelihood_tables.items():
    #     print("Likelihood Table for Feature:", feature)
    #     for label, probabilities in table.items():
    #         print("Class Label:", label)
    #         for value, probability in probabilities.items():
    #             print(f"    {value}: {probability}")
    #         print()
    #     print()

    return likelihood_tables

likelihood_tables = compute_likelihood_probabilities(data)


def predict(sample, likelihood_tables):
    probabilities_yes = {}
    probabilities_no = {}


    # Iterate over each class label
    for label in ['Yes', 'No']:
        probability = 1

        # Compute the product of conditional probabilities for each feature given the class label
        for feature, value in sample.items():
            # Check if the value is present in the likelihood table
            if value in likelihood_tables[feature][label]:
                probability *= likelihood_tables[feature][label][value]
            else:
                # If the value is unseen, assign 0 (ex : there is no way its overcast and NO)
                probability *= 0

        if label == 'Yes':
            probabilities_yes[label] = probability
        else:
            probabilities_no[label] = probability

    if probabilities_yes['Yes'] > probabilities_no['No']:
        return 'Yes'
    else:
        return 'No'

test_sample = {'Outlook': 'Overcast', 
               'Temperature': 'Mild', 
               'Humidity': 'High',
               'Wind': 'Strong'}

prediction = predict(test_sample, likelihood_tables)
print("Prediction:", prediction)
