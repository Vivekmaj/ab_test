#import packages
import pandas as pd
import numpy as np
import datetime
from scipy.stats import chi2_contingency, beta
import streamlit as st
import streamlit.components.v1 as stc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

html_temp = """
		<div style="background-color:tomato;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">A/B Testing App </h1>
		</div>
		"""

def main():
    stc.html(html_temp)

    menu = ["Home", 'Bayesian A/B Test', 'Frequentist A/B Test']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader('A/B Testing Introduction')
        st.write('A/B testing is one of the most important tools for optimizing most things we interact with on our computers, phones and tablets. From website layouts to social media ads and product features, every button, banner and call to action has probably been A/B tested. And these tests can be extremely granular; Google famously tested "40 shades of blue" to decide what shade of blue should be used for links on the Google and Gmail landing pages.')
        st.markdown("""**Note**: The link to the data source can be found [here](https://www.kaggle.com/zhangluyuan/ab-testing)""")

        st.write(""" ### Data Info
        We developed a new webpage and want to test it's effects on purchase conversion. As such we split our users evenly into 2 groups:
            - Control: They get the old webpage.
            - Treatment: They get the new webpage. 
            
        We have 3 weeks of logged exposure/conversion data. Let's define these terms:
            - Exposure: A user is bucketed as control or treatment and sees their corresponding page for the first time in the experiment duration.
            - Conversion: An exposed user makes a purchase within 7 days of being first exposed. 
            
        Metric to track:
            - Purchase Conversion: # Converted Users / # Exposed Users""")

    if choice == 'Bayesian A/B Test':

        st.header('Bayesian Section')

        st.subheader('Demo Data')
        #import data
        raw_data = pd.read_csv("ab_data.csv")
        df = raw_data.copy()

        st.write("Number of rows: ", df.shape[0], " Number of columns: ", df.shape[1])
        st.write(df.head())

        with st.expander('View Basic Statistics'):
            start_time = datetime.datetime.strptime(df['timestamp'].min(), '%Y-%m-%d %H:%M:%S.%f')
            end_time = datetime.datetime.strptime(df['timestamp'].max(), '%Y-%m-%d %H:%M:%S.%f')
            data_duration = (end_time - start_time).days

            st.write('Number of unique users in experiment:')
            st.write(df['user_id'].nunique())

            st.write('Data collection period (days):')
            st.write(data_duration)

            st.write('Landing pages to compare:')
            st.write(df['landing_page'].unique().tolist())

            st.write('Users in control group (%):')
            st.write(round(df[df['group']=='control'].shape[0] * 100 / df.shape[0]))

        sample = df[df['user_id'].isin([746755,722274])]
        first_conversion = sample.groupby('user_id')['timestamp'].min().to_frame().reset_index()
        sample = sample.merge(first_conversion, on=['user_id', 'timestamp'])

        counter = df['user_id'].value_counts()

        valid_users = pd.DataFrame(counter[counter == 1].index, columns=['user_id'])
        df = df.merge(valid_users, on=['user_id'])

        df['week'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').isocalendar()[1])

        st.subheader('Results Section')

        period = st.slider('Results Period (Weeks):', 2, 5)
        lift_perc = st.number_input('Insert Lift (%) to Investigate:')

        with st.expander('View Results'):
            prior = df[(df['week'] == 1) & (df['group']=='control')]
            prior_means = []
            for i in range(10000):
                prior_means.append(prior.sample(1000)['converted'].mean())

            prior_alpha, prior_beta, _, _ = beta.fit(prior_means, floc=0, fscale=1)

            # Get Stats
            NUM_WEEKS = period # Vary number to get experiment data at weekly points in time
            experiment_data = df[(df['week'] > 1) & (df['week'] <= NUM_WEEKS)]
            control = experiment_data[experiment_data['group']=='control']
            treatment = experiment_data[experiment_data['group']=='treatment']

            control_conversion_perc = round(control['converted'].sum() * 100/ control['converted'].count(), 3)
            treatment_conversion_perc = round(treatment['converted'].sum() * 100/ treatment['converted'].count(), 3)
            lift = round((treatment_conversion_perc - control_conversion_perc) / control_conversion_perc , 3)

            st.write('Treatment Conversion Rate (%):')
            st.write(treatment_conversion_perc)

            st.write('Control Conversion Rate (%):')
            st.write(control_conversion_perc)

            st.write('Lift (%):')
            st.write(lift)

            control_converted = control['converted'].sum()
            treatment_converted = treatment['converted'].sum()
            control_non_converted = control['converted'].count() - control_converted
            treatment_non_converted = treatment['converted'].count() - treatment_converted

            # Update Prior parameters with experiment conversion rates
            posterior_control = beta(prior_alpha + control_converted, prior_beta + control_non_converted)
            posterior_treatment = beta(prior_alpha + treatment_converted, prior_beta + treatment_non_converted)

            # Sample from Posteriors
            control_samples = posterior_control.rvs(1000)
            treatment_samples = posterior_treatment.rvs(1000)
            probability = np.mean(treatment_samples > control_samples)
            
            st.write('% probability treatment > control:')
            st.write(round(probability * 100, 2))

            lift_percentage = (treatment_samples - control_samples) / control_samples
            st.write(f'Probability of seeing a {lift_perc}% lift:')
            st.write(str(np.mean((100 * lift_percentage) > lift_perc) * 100) + '%')

    if choice == 'Frequentist A/B Test':

        st.header('Frequentist Section')

        st.write(""" ### Defining Hypothesis
        
        H0: Control & Treatment are independent.
        H1: Control & Treatment are not independent.""")

        st.subheader('Demo Data')
        #import data
        raw_data = pd.read_csv("ab_data.csv")
        df = raw_data.copy()

        st.write("Number of rows: ", df.shape[0], " Number of columns: ", df.shape[1])
        st.write(df.head())

        with st.expander('View Basic Statistics'):
            start_time = datetime.datetime.strptime(df['timestamp'].min(), '%Y-%m-%d %H:%M:%S.%f')
            end_time = datetime.datetime.strptime(df['timestamp'].max(), '%Y-%m-%d %H:%M:%S.%f')
            data_duration = (end_time - start_time).days

            st.write('Number of unique users in experiment:')
            st.write(df['user_id'].nunique())

            st.write('Data collection period (days):')
            st.write(data_duration)

            st.write('Landing pages to compare:')
            st.write(df['landing_page'].unique().tolist())

            st.write('Users in control group (%):')
            st.write(round(df[df['group']=='control'].shape[0] * 100 / df.shape[0]))

        sample = df[df['user_id'].isin([746755,722274])]
        first_conversion = sample.groupby('user_id')['timestamp'].min().to_frame().reset_index()
        sample = sample.merge(first_conversion, on=['user_id', 'timestamp'])

        counter = df['user_id'].value_counts()

        valid_users = pd.DataFrame(counter[counter == 1].index, columns=['user_id'])
        df = df.merge(valid_users, on=['user_id'])

        df['week'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').isocalendar()[1])
        
        st.subheader('Results Section')

        period = st.slider('Results Period (Weeks):', 2, 5)

        with st.expander('View Results'):
            # Get Stats
            NUM_WEEKS = period # Vary number to get experiment data at weekly points in time
            experiment_data = df[df['week'] <= NUM_WEEKS]
            control = experiment_data[experiment_data['group']=='control']
            treatment = experiment_data[experiment_data['group']=='treatment']

            control_conversion_perc = round(control['converted'].sum() * 100/ control['converted'].count(), 3)
            treatment_conversion_perc = round(treatment['converted'].sum() * 100/ treatment['converted'].count(), 3)
            lift = round(treatment_conversion_perc - control_conversion_perc, 3)

            st.write('Treatment Conversion Rate (%):')
            st.write(treatment_conversion_perc)

            st.write('Control Conversion Rate (%):')
            st.write(control_conversion_perc)

            st.write('Lift (%):')
            st.write(lift)

            # Create Contingency Table for Chi Squared Test
            control_converted = control['converted'].sum()
            treatment_converted = treatment['converted'].sum()
            control_non_converted = control['converted'].count() - control_converted
            treatment_non_converted = treatment['converted'].count() - treatment_converted
            contingency_table = np.array([[control_converted, control_non_converted], 
                                        [treatment_converted, treatment_non_converted]])

            chi, p_value, _, _ = chi2_contingency(contingency_table, correction=False)

            if p_value < 0.05:
                st.write('P-Value:')
                st.write(p_value)
                st.write('Verdict:')
                st.info('Sufficient evidence to reject H0')

            else:
                st.write('P-Value:')
                st.write(p_value)
                st.write('Verdict:')
                st.info('Fail to reject H0')

if __name__ == '__main__':
    main()