import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf  # Import the numpy_financial library

# ------------------------------
# Set Streamlit to wide mode
# ------------------------------
st.set_page_config(layout="wide")

# ------------------------------
# Main Title and Description
# ------------------------------
st.title("Spending and Surplus Simulation with Mortality Probability")
st.write("""
This app allows you to adjust various parameters like fixed payments, income targets, interest rates, 
and more, and then run simulations to see how different scenarios affect spending and surplus over time.
Additionally, it calculates the probability that either a male or female (both aged 65) will be alive after the selected number of years.
""")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data(cpi_file_path, life_table_file_path):
    # Load CPI end value calculations
    excel_data = pd.read_excel(cpi_file_path, header=None)
    
    # Load and clean the life table for mortality probability
    life_table = pd.read_excel(life_table_file_path)
    
    # Clean the life table by selecting relevant columns and renaming them
    life_table_clean = life_table.iloc[2:].rename(columns={
        'Unnamed: 0': 'Exact age',
        'Unnamed: 2': 'Male Number of Lives',
        'Unnamed: 5': 'Female Number of Lives'
    })
    
    # Convert columns to numeric
    life_table_clean['Exact age'] = pd.to_numeric(life_table_clean['Exact age'], errors='coerce')
    life_table_clean['Male Number of Lives'] = pd.to_numeric(life_table_clean['Male Number of Lives'], errors='coerce')
    life_table_clean['Female Number of Lives'] = pd.to_numeric(life_table_clean['Female Number of Lives'], errors='coerce')
    
    return excel_data, life_table_clean

# Paths to your data files
cpi_file_path = 'cpi_end_val_calcs.xlsx'  # Update with your actual file path
life_table_file_path = 'life_table.xlsx'  # Update with your actual file path

# Load the data
excel_data, life_table_clean = load_data(cpi_file_path, life_table_file_path)

# ------------------------------
# Define Functions
# ------------------------------
def survival_probability(sex, current_age, years_to_project):
    target_age = current_age + years_to_project
    if current_age not in life_table_clean['Exact age'].values or target_age not in life_table_clean['Exact age'].values:
        return f"Data not available for the age range {current_age} to {target_age}."
    
    # Get the number of lives at the current age and target age
    if sex.lower() == 'male':
        lives_at_current_age = life_table_clean[life_table_clean['Exact age'] == current_age]['Male Number of Lives'].values[0]
        lives_at_target_age = life_table_clean[life_table_clean['Exact age'] == target_age]['Male Number of Lives'].values[0]
    elif sex.lower() == 'female':
        lives_at_current_age = life_table_clean[life_table_clean['Exact age'] == current_age]['Female Number of Lives'].values[0]
        lives_at_target_age = life_table_clean[life_table_clean['Exact age'] == target_age]['Female Number of Lives'].values[0]
    else:
        return "Invalid sex entered. Please enter 'male' or 'female'."
    
    # Calculate the probability of survival
    prob_alive = lives_at_target_age / lives_at_current_age
    return prob_alive

def probability_either_alive(sex1, age1, sex2, age2, years_to_project):
    prob1 = survival_probability(sex1, age1, years_to_project)
    
    if sex2 and age2:  # If there is a second person
        prob2 = survival_probability(sex2, age2, years_to_project)
        
        if isinstance(prob1, str) or isinstance(prob2, str):
            return f"Error: {prob1} {prob2}"
        
        # Probability that either is alive (1 - both are not alive)
        prob_either_alive = 1 - (1 - prob1) * (1 - prob2)
        return prob_either_alive
    else:
        return prob1  # If there's only one person, return their probability

def calculate_inflation_rates(cpi_factors):
    inflation_rates = []
    for i in range(len(cpi_factors)):
        if i == 0:
            inflation_rates.append(0)  # No inflation for the first year
        else:
            inflation_rate = (cpi_factors[i] / cpi_factors[i - 1]) - 1
            inflation_rates.append(inflation_rate)
    return inflation_rates

def calculate_spending_and_surplus(cpi_factors, income_target, fixed_payment, interest_rate, surplus_threshold):
    prior_surplus = 0
    total_actual_spend = 0
    year_results = []

    # Calculate annual inflation rates
    inflation_rates = calculate_inflation_rates(cpi_factors)

    # Initialize the inflation-adjusted target for the first year
    iat = income_target

    for year in range(num_years):
        prior_surplus *= (1 + interest_rate)

        # Get CPI Factor with 4 decimal places
        cpi_factor = round(cpi_factors[year], 4)

        if year > 0:
            iat *= (1 + inflation_rates[year])

        iatss = iat - income_target
        as_val = iat - iatss
        surplus = fixed_payment + iatss - as_val + prior_surplus

        if surplus < 0:
            surplus = 0
            as_val = iat

        # If the surplus is greater than the threshold, add the extra to actual spend
        if surplus > surplus_threshold:
            extra_spend = surplus - surplus_threshold
            as_val += extra_spend
            surplus = surplus_threshold

        total_actual_spend += as_val

        year_results.append({
            'Year': year + 1,
            'Inflation Rate (%)': round(inflation_rates[year] * 100, 2),
            'CPI Factor': f"{cpi_factor:.4f}",  # CPI Factor with 4 decimals
            'Inflation Adjusted Target': round(iat),
            'Inflation Adjusted Target Surplus/Shortfall': round(iatss),
            'Actual Spend': round(as_val),
            'Ending Surplus': round(surplus)
        })

        prior_surplus = surplus

    average_actual_spend = round(total_actual_spend / num_years)
    ending_surplus = round(prior_surplus)

    return year_results, average_actual_spend, ending_surplus

def calculate_last_n_years_success(last_n_years_df, income_target):
    # Group by the 'Row' to calculate average actual spend for each trial in the last N years
    last_n_years_summary = last_n_years_df.groupby('Row').agg({'Actual Spend': 'mean'}).reset_index()

    # Calculate success rate: percentage of trials where the average actual spend >= income target
    successful_trials_last_n = (last_n_years_summary['Actual Spend'] >= income_target).sum()
    total_trials_last_n = len(last_n_years_summary)

    if total_trials_last_n > 0:
        success_rate_last_n = (successful_trials_last_n / total_trials_last_n) * 100
    else:
        success_rate_last_n = 0

    return round(success_rate_last_n, 2)

def calculate_irr(actual_spend, ending_surplus, portfolio_value):
    # Cash flow: initial outlay from portfolio_value, followed by positive cash inflows (spending), and ending surplus at the end
    cash_flows = [-portfolio_value] + [spend for spend in actual_spend] + [ending_surplus]
    
    # Calculate IRR using numpy_financial's irr function
    irr_value = npf.irr(cash_flows)
    
    # Return IRR as percentage
    if irr_value is not None and not np.isnan(irr_value):
        return round(irr_value * 100, 2)  # Convert IRR to percentage and round to 2 decimals
    else:
        return None

def calculate_irr_fixed_payment(fixed_payment, portfolio_value, num_years, ending_surplus):
    # Cash flows: Initial outlay (negative) followed by Fixed Payment cash flows for N years, and then the ending surplus
    cash_flows = [-portfolio_value] + [fixed_payment] * num_years + [ending_surplus]
    
    # Calculate IRR using numpy_financial's irr function
    irr_value = npf.irr(cash_flows)
    
    # Return IRR as percentage
    if irr_value is not None and not np.isnan(irr_value):
        return round(irr_value * 100, 2)  # Convert IRR to percentage and round to 2 decimals
    else:
        return None

# ------------------------------
# Sidebar for User Inputs
# ------------------------------
import streamlit as st


# Sidebar for User Inputs
st.sidebar.header("**Simulation Parameters**")

# Financial Inputs
st.sidebar.subheader("Financial Parameters")

# Portfolio Value input and display as currency (no decimals)
portfolio_value = st.sidebar.number_input("Portfolio Value (Upfront Outlay)", value=1000000, step=1000)
st.sidebar.write(f"Portfolio Value: ${portfolio_value:,.0f}")

# Fixed Payment input and display as currency (no decimals)
fixed_payment = st.sidebar.number_input("Fixed Payment", value=65000, step=500)
st.sidebar.write(f"Fixed Payment: ${fixed_payment:,.0f}")

# Income Target input and display as currency (no decimals)
income_target = st.sidebar.number_input("Income Target", value=40000, step=500)
st.sidebar.write(f"Income Target: ${income_target:,.0f}")

# Surplus Threshold input and display as currency (no decimals)
surplus_threshold = st.sidebar.number_input("Surplus Threshold", value=100000, min_value=0, step=1000)
st.sidebar.write(f"Surplus Threshold: ${surplus_threshold:,.0f}")
num_years = st.sidebar.number_input("Number of Years", value=24, min_value=1, max_value=50)
interest_rate = st.sidebar.number_input("Interest Rate (e.g., 0.03 for 3%)", value=0.03, format="%.4f")

# Mortality Inputs
st.sidebar.subheader("Mortality Parameters")
num_people = st.sidebar.radio("Number of People", ("One", "Two"), index=1)  # Set default to "Two"
age1 = st.sidebar.number_input("Current Age of Person 1", value=65, min_value=0, max_value=120)
sex1 = st.sidebar.selectbox("Sex of Person 1", ("Male", "Female"), index=0)  # Set default to "Male"

# If two people are selected, show additional inputs
if num_people == "Two":
    age2 = st.sidebar.number_input("Current Age of Person 2", value=65, min_value=0, max_value=120)
    sex2 = st.sidebar.selectbox("Sex of Person 2", ("Male", "Female"), index=1)  # Set default to "Female"
else:
    age2 = None
    sex2 = None

# Analysis Options
st.sidebar.subheader("Analysis Options")
n_years = st.sidebar.slider("Select Last N Years to focus on latter part of retirement", min_value=1, max_value=num_years, value=5)

# Display Options
st.sidebar.subheader("Display Options")
show_success_rate = st.sidebar.checkbox("Show Probability of Success", value=True)
show_detailed_results = st.sidebar.checkbox("Show Detailed Results", value=False)
show_summary_results = st.sidebar.checkbox("Show Summary Results", value=False)
show_percentile_stats = st.sidebar.checkbox("Show Percentile Statistics", value=True)
show_last_n_years_percentile_stats = st.sidebar.checkbox(f"Show Last {n_years} Years Percentile Statistics", value=True)
show_irr_stats = st.sidebar.checkbox("Show IRR Statistics", value=True)


# ------------------------------
# Main Area - Calculations and Outputs
# ------------------------------

# Calculate and display the probability that either person will be alive
either_alive_prob = probability_either_alive(sex1, age1, sex2, age2, num_years)

if isinstance(either_alive_prob, str):
    st.error(either_alive_prob)
else:
    if num_people == "Two":
        # Output for two people
        st.write(f"Probability that either ({sex1}, age {age1}) or ({sex2}, age {age2}) will be alive after {num_years} years: {either_alive_prob:.0%}")
        
        # Output for the inverse probability (1 - prob of either living)
        st.write(f"That means there is a {1 - either_alive_prob:.0%} chance that neither ({sex1}, age {age1}) nor ({sex2}, age {age2}) will be alive after {num_years} years.")
    else:
        # Output for one person
        st.write(f"Probability that ({sex1}, age {age1}) will be alive after {num_years} years is approximately: {either_alive_prob:.0%}")
        
        # Output for the inverse probability (1 - prob of living)
        st.write(f"That means there is a {1 - either_alive_prob:.0%} chance that ({sex1}, age {age1}) will not be alive after {num_years} years.")



# Calculate spending and surplus
all_rows = []
summary_rows = []
successful_trials = 0

for index, row in excel_data.iterrows():
    cpi_factors = row[1:num_years + 1].values

    if not pd.isnull(cpi_factors).any():
        spending_and_surplus, average_actual_spend, ending_surplus = calculate_spending_and_surplus(
            cpi_factors, income_target, fixed_payment, interest_rate, surplus_threshold)

        for result in spending_and_surplus:
            result['Row'] = index + 1
            all_rows.append(result)

        summary_rows.append({
            'Row': index + 1,
            'Average Actual Spend': average_actual_spend,
            'Ending Surplus': ending_surplus
        })

        if average_actual_spend >= income_target:
            successful_trials += 1

total_trials = len(summary_rows)
if total_trials > 0:
    success_rate = round((successful_trials / total_trials) * 100)  # Round success rate to nearest whole number
else:
    success_rate = 0

# Convert results to DataFrames
detailed_results_df = pd.DataFrame(all_rows).round(0)
summary_results_df = pd.DataFrame(summary_rows).round(0)

# Isolate the last N years for percentile calculations
last_n_years_df = detailed_results_df[detailed_results_df['Year'] > (num_years - n_years)]

# Calculate success rate for the last N years
success_rate_last_n_years = calculate_last_n_years_success(last_n_years_df, income_target)

# Display success rate for last N years
if show_success_rate:
    st.write(f"Probability of Success Complete Period: {int(success_rate)}%")
    st.write(f"Probability of Success for Last {n_years} Years: {int(success_rate_last_n_years)}%")

# Display Detailed Results
if show_detailed_results:
    st.subheader("Detailed Results")
    st.dataframe(detailed_results_df)

# Display Summary Results
if show_summary_results:
    st.subheader("Summary Results")
    st.dataframe(summary_results_df)

# Calculate and Display Percentile Statistics
if show_percentile_stats:
    summary_stats = {
        'Min': [
            summary_results_df['Average Actual Spend'].min(),
            summary_results_df['Ending Surplus'].min()
        ],
        '10th Percentile': [
            summary_results_df['Average Actual Spend'].quantile(0.10),
            summary_results_df['Ending Surplus'].quantile(0.10)
        ],
        '25th Percentile': [
            summary_results_df['Average Actual Spend'].quantile(0.25),
            summary_results_df['Ending Surplus'].quantile(0.25)
        ],
        'Median': [
            summary_results_df['Average Actual Spend'].median(),
            summary_results_df['Ending Surplus'].median()
        ],
        '75th Percentile': [
            summary_results_df['Average Actual Spend'].quantile(0.75),
            summary_results_df['Ending Surplus'].quantile(0.75)
        ],
        '90th Percentile': [
            summary_results_df['Average Actual Spend'].quantile(0.90),
            summary_results_df['Ending Surplus'].quantile(0.90)
        ],
        'Max': [
            summary_results_df['Average Actual Spend'].max(),
            summary_results_df['Ending Surplus'].max()
        ]
    }
    
    # Create a DataFrame for percentile statistics
    percentile_df = pd.DataFrame(summary_stats, index=["Average Actual Spend", "Ending Surplus"]).round(0)

    # Calculate the percentage of the income target
    percentage_stats = percentile_df.loc["Average Actual Spend"].apply(lambda x: (x / income_target) * 100)

    # Add the percentage row to the DataFrame
    percentile_df.loc['As % of Income Target'] = percentage_stats.apply(lambda x: f"{x:.0f}%")

    # Check if any values are below 100%
    below_100 = percentage_stats[percentage_stats < 100]

    # If there are values below 100%, calculate Budget Cut % and add the row
    if not below_100.empty:
        budget_cut_stats = below_100.apply(lambda x: f"{100 - x:.0f}%")
        percentile_df.loc['Budget Cut %'] = budget_cut_stats
        # Format Budget Cut % as percentage
        percentile_df.loc['Budget Cut %'] = budget_cut_stats.apply(lambda x: x)
    
    # Format values as currency (no decimals) except for percentage and budget cut
    percentile_df = percentile_df.applymap(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x)
    
    # Display the percentile statistics
    st.subheader("Percentile Statistics")
    st.write("The following amounts are in real terms or 'Today's Dollars'.")
    st.dataframe(percentile_df)


# Calculate and Display Percentile Statistics for Last N Years
if show_last_n_years_percentile_stats:
    last_n_years_stats = {
        'Min': [
            last_n_years_df['Actual Spend'].min(),
            last_n_years_df['Ending Surplus'].min()
        ],
        '10th Percentile': [
            last_n_years_df['Actual Spend'].quantile(0.10),
            last_n_years_df['Ending Surplus'].quantile(0.10)
        ],
        '25th Percentile': [
            last_n_years_df['Actual Spend'].quantile(0.25),
            last_n_years_df['Ending Surplus'].quantile(0.25)
        ],
        'Median': [
            last_n_years_df['Actual Spend'].median(),
            last_n_years_df['Ending Surplus'].median()
        ],
        '75th Percentile': [
            last_n_years_df['Actual Spend'].quantile(0.75),
            last_n_years_df['Ending Surplus'].quantile(0.75)
        ],
        '90th Percentile': [
            last_n_years_df['Actual Spend'].quantile(0.90),
            last_n_years_df['Ending Surplus'].quantile(0.90)
        ],
        'Max': [
            last_n_years_df['Actual Spend'].max(),
            last_n_years_df['Ending Surplus'].max()
        ]
    }
    
    # Create a DataFrame for the last N years' percentile statistics
    last_n_years_percentile_df = pd.DataFrame(last_n_years_stats, index=["Actual Spend (Last N Years)", "Ending Surplus (Last N Years)"]).round(0)

    # Calculate the percentage of the income target
    percentage_stats_last_n_years = last_n_years_percentile_df.loc["Actual Spend (Last N Years)"].apply(lambda x: (x / income_target) * 100)

    # Add the percentage row to the DataFrame
    last_n_years_percentile_df.loc['As % of Income Target'] = percentage_stats_last_n_years.apply(lambda x: f"{x:.0f}%")

    # Check if any values are below 100%
    below_100_last_n_years = percentage_stats_last_n_years[percentage_stats_last_n_years < 100]

    # If there are values below 100%, calculate Budget Cut % and add the row
    if not below_100_last_n_years.empty:
        budget_cut_stats_last_n_years = below_100_last_n_years.apply(lambda x: f"{100 - x:.0f}%")
        last_n_years_percentile_df.loc['Budget Cut %'] = budget_cut_stats_last_n_years
    
    # Format values as currency (no decimals) except for percentage and budget cut
    last_n_years_percentile_df = last_n_years_percentile_df.applymap(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x)
    
    # Display the last N years' percentile statistics
    st.subheader(f"Percentile Statistics for Last {n_years} Years")
    st.write("The following amounts are in real terms or 'Today's Dollars'.")
    st.dataframe(last_n_years_percentile_df)



# ------------------------------
# IRR Calculations and Outputs
# ------------------------------

if show_irr_stats:
    detailed_results_with_irr = []
    for trial_row in summary_rows:
        # Get actual spending for the trial from detailed results
        actual_spend = detailed_results_df[detailed_results_df['Row'] == trial_row['Row']]['Actual Spend'].values.tolist()
        ending_surplus = trial_row['Ending Surplus']
        
        # Calculate IRR for the trial using Portfolio Value
        irr_value = calculate_irr(actual_spend, ending_surplus, portfolio_value)
        
        # Append IRR to the trial summary
        trial_row['IRR (%)'] = irr_value  # Keep IRR as a numeric value for calculations
        detailed_results_with_irr.append(trial_row)
    
    # Convert the results with IRR to a DataFrame for display
    summary_with_irr_df = pd.DataFrame(detailed_results_with_irr).round(2)
    
    # Display the updated summary with IRR (formatted only for display)
    if show_summary_results:
        st.subheader("Summary Results with IRR")
        # Format IRR values to two decimal places (for display)
        summary_with_irr_df_display = summary_with_irr_df.copy()
        summary_with_irr_df_display['IRR (%)'] = summary_with_irr_df_display['IRR (%)'].map(lambda x: f"{x:.2f}%")
        st.dataframe(summary_with_irr_df_display)
    
    # Calculate percentile statistics for IRR
    if not summary_with_irr_df.empty:
        irr_stats = {
            'Min': [summary_with_irr_df['IRR (%)'].min()],
            '10th Percentile': [summary_with_irr_df['IRR (%)'].quantile(0.10)],
            '25th Percentile': [summary_with_irr_df['IRR (%)'].quantile(0.25)],
            'Median': [summary_with_irr_df['IRR (%)'].median()],
            '75th Percentile': [summary_with_irr_df['IRR (%)'].quantile(0.75)],
            '90th Percentile': [summary_with_irr_df['IRR (%)'].quantile(0.90)],
            'Max': [summary_with_irr_df['IRR (%)'].max()]
        }
    
        # Create a DataFrame for IRR statistics
        irr_stats_df = pd.DataFrame(irr_stats, index=["IRR (%)"]).round(2)
        
        # Format IRR statistics as percentage for display
        irr_stats_df = irr_stats_df.applymap(lambda x: f"{x:.2f}%")
    
        # Display the IRR statistics
        st.subheader("IRR Statistics")
        # st.dataframe(irr_stats_df)
        
        # Calculate IRR for Fixed Payment over N years and display after subheader
        if len(summary_rows) > 0:
            irr_fixed_payment = calculate_irr_fixed_payment(
                fixed_payment, 
                portfolio_value, 
                num_years, 
                summary_rows[0]['Ending Surplus']  # Using the first trial's Ending Surplus for now
            )
            st.write(f"IRR for Fixed Payment over {num_years} years: {irr_fixed_payment:.2f}%")
        else:
            st.write(f"IRR for Fixed Payment over {num_years} years: N/A")
         
         
if isinstance(either_alive_prob, str):
    st.error(either_alive_prob)
else:
    if num_people == "Two":
        # Output for two people
        # st.write(f"Probability that either ({sex1}, age {age1}) or ({sex2}, age {age2}) will be alive after {num_years} years: {either_alive_prob:.0%}")
        
        # Output for the inverse probability (1 - prob of either living)
        st.write(f"Remember, there is a {1 - either_alive_prob:.0%} chance that neither ({sex1}, age {age1}) nor ({sex2}, age {age2}) will be alive after {num_years} years. If so, then the returns would be lower than shown below.")
    else:
        # Output for one person
        # st.write(f"Probability that ({sex1}, age {age1}) will be alive after {num_years} years is approximately: {either_alive_prob:.0%}")
        
        # Output for the inverse probability (1 - prob of living)
        st.write(f"Remember, there is a {1 - either_alive_prob:.0%} chance that ({sex1}, age {age1}) will not be alive after {num_years} years. If so, then the returns would be lower than shown below.")   
                
    # Calculate percentile statistics for IRR
    if not summary_with_irr_df.empty:
        irr_stats = {
            'Min': [summary_with_irr_df['IRR (%)'].min()],
            '10th Percentile': [summary_with_irr_df['IRR (%)'].quantile(0.10)],
            '25th Percentile': [summary_with_irr_df['IRR (%)'].quantile(0.25)],
            'Median': [summary_with_irr_df['IRR (%)'].median()],
            '75th Percentile': [summary_with_irr_df['IRR (%)'].quantile(0.75)],
            '90th Percentile': [summary_with_irr_df['IRR (%)'].quantile(0.90)],
            'Max': [summary_with_irr_df['IRR (%)'].max()]
        }
    
        # Create a DataFrame for IRR statistics
        irr_stats_df = pd.DataFrame(irr_stats, index=["IRR (%)"]).round(2)
        
        # Format IRR statistics as percentage for display
        irr_stats_df = irr_stats_df.applymap(lambda x: f"{x:.2f}%")
    
        # Display the IRR statistics
        # st.subheader("IRR Statistics")
        st.dataframe(irr_stats_df)

        
# ------------------------------
# Sidebar Download Buttons
# ------------------------------
st.sidebar.header("**Download Results**")

# Download Detailed Results
if not detailed_results_df.empty:
    st.sidebar.download_button(
        label="Download Detailed Results as CSV",
        data=detailed_results_df.to_csv(index=False),
        file_name="detailed_spending_surplus.csv",
        key="download_detailed_results"
    )
else:
    st.sidebar.write("No Detailed Results available to download.")

# Download Summary Results
if not summary_results_df.empty:
    st.sidebar.download_button(
        label="Download Summary Results as CSV",
        data=summary_results_df.to_csv(index=False),
        file_name="summary_spending_surplus.csv",
        key="download_summary_results"
    )
else:
    st.sidebar.write("No Summary Results available to download.")

# Download Summary Results with IRR
if show_irr_stats and 'summary_with_irr_df' in locals() and not summary_with_irr_df.empty:
    st.sidebar.download_button(
        label="Download Summary Results with IRR as CSV",
        data=summary_with_irr_df.to_csv(index=False),
        file_name="summary_with_irr_spending_surplus.csv",
        key="download_summary_with_irr"
    )
else:
    st.sidebar.write("No IRR Summary Results available to download.")
