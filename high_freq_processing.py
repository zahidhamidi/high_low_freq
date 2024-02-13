import csv
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import seaborn as sns

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

# Specify the path to your CSV file
csv_file_path = r"C:\Users\ZHamid2\OneDrive - SLB\Downloads\Parameters_ 15H.csv"

st.title("Drilling Parameter High-Frequency Sensor Data Streamlined Processing")
st.write("Automated Workflow for High-frequency to Low-frequency transformation of Sensor Data")
st.divider()

upload_file = st.file_uploader("Upload your file here")
st.divider()

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
first_original_rows = len(df)
    
    
st.subheader("Data Audit : Completeness Check",divider=True)

# Check data completeness
header_list = df.columns

# Creating two columns in Streamlit
col1, col2 = st.columns(2)


st.write("Checking data completeness...\n")
st.write(f"Total rows count : {len(df)-1}")


# Initialize the checklist_data with False for each column
checklist_data = [False] * len(header_list)
null_counts = [0] * len(header_list)  # Initialize a list to store null counts

point = 0

# Iterate through the headers
for i, header in enumerate(header_list):
    header_lower = header.lower()

    if header_lower in ["hole depth", "depth"]:
        checklist_data[i] = True
        point += 1

    elif header_lower == "bit depth":
        checklist_data[i] = True
        point += 1

    elif header_lower in ["rotary rpm", "rpm"]:
        checklist_data[i] = True
        point += 1

    elif header_lower in ["weight on bit", "wob"]:
        checklist_data[i] = True
        point += 1

    elif header_lower in ["rate of penetration", "rop"]:
        checklist_data[i] = True
        point += 1

    elif header_lower in ["total pump output", "pump"]:
        checklist_data[i] = True
        point += 1

    elif header_lower == "hh:mm:ss":
        checklist_data[i] = True
        point += 1

    # Count non-null values in the respective column
    null_counts[i] = len(df) - df[header].count()

# Filter the headers and checklist_data to show only where the checkbox is True
filtered_data = [(header, exists, null_count) for header, exists, null_count in zip(header_list, checklist_data, null_counts)]

# Create a DataFrame for the existing data
df_new = pd.DataFrame({
    "Headers": [header for header, _, _ in filtered_data],
    "Exists": [exists for _, exists, _ in filtered_data],
    "Null rows count": [null_count for _, _, null_count in filtered_data]
})


# Create a bar chart using Plotly Express
fig = px.bar(df_new, x="Headers", y="Exists", text="Null rows count",
            hover_data=["Headers", "Null rows count"],
            labels={"Exists": "Existence Points"},
            color_discrete_sequence=['blue'])  # Adjust color as needed

# Update layout for better readability
fig.update_layout(
    title='Header Existence and Null Counts',
    xaxis_title='Headers',
    yaxis_title='Existence Points',
)

# Display the Plotly bar chart in the Streamlit app
st.plotly_chart(fig, use_container_width=True)






# Display the total number of existing columns
st.write(f"{point} out of 7 main data columns exist.")

st.divider()

st.subheader("Operation Parameter Plot : Cleaned",divider=True)


# Assuming df is your DataFrame and 'Hole Depth' is the column with depth values
df['Drilling Progression'] = df['Hole Depth'].diff()

# Filter the DataFrame to include only rows with 'Drilling Progression' > 0
df = df[df['Drilling Progression'] > 0]

# Remove duplicates based on specified columns
df = df.drop_duplicates(subset=['Total Pump Output', 'Top Drive Torque', 'Standpipe Pressure', 'Hook Load', 'Rate Of Penetration', 'Weight on Bit'], keep='first')

# Convert 'Time Of Penetration' to datetime
df['Time Of Penetration'] = pd.to_datetime(df['Time Of Penetration'])

# Remove rows based on conditions
condition = (df.get('Rotary RPM', 0) > 500) & \
            (df.get('Weight on Bit', 0) > 300) & \
            (df.get('Rate Of Penetration', 0) > 500) & \
            (df.get('Total Pump Output', 0) > 500)

df = df[~condition]




# Filter the DataFrame : Hole depth > 30
start_hole_depth = 30
df = df[df['Hole Depth'] > start_hole_depth]

hole_size = ['All',17.5,16,12.25,8.5,6.125]


original_rows = len(df)


# Apply conditions using if-else statements
for index, row in df.iterrows():
    

    
    if 63.5 <= row['Hole Depth'] <= 65:
        df.at[index, 'Hole Diameter'] = 17.5
        df.at[index, 'Run Number'] = 1
    
    elif 65 <= row['Hole Depth'] <= 1301:
        df.at[index, 'Hole Diameter'] = 16
        df.at[index, 'Run Number'] = 2
    
    elif 1301 <= row['Hole Depth'] <= 2704:
        df.at[index, 'Hole Diameter'] = 12.25
        df.at[index, 'Run Number'] = 3

    elif 2704 <= row['Hole Depth'] <= 3430:
        df.at[index, 'Hole Diameter'] = 8.5
        df.at[index, 'Run Number'] = 4
    
    elif 3430 <= row['Hole Depth'] <= 3433:
        df.at[index, 'Hole Diameter'] = 6.125
        df.at[index, 'Run Number'] = 5

    elif 3433 <= row['Hole Depth'] <= 3446:
        df.at[index, 'Hole Diameter'] = 6.125
        df.at[index, 'Run Number'] = 6
    
    elif 3446 <= row['Hole Depth'] <= 3677:
        df.at[index, 'Hole Diameter'] = 6.125
        df.at[index, 'Run Number'] = 7






# Move the 'Drilling Progression' column to index 1
df.insert(1, 'Drilling Progression', df.pop('Drilling Progression'))

# Move the 'Drilling Progression' column to index 1
df.insert(0, 'Hole Diameter', df.pop('Hole Diameter'))

# Convert the 'Timestamp' column to datetime format
df['HH:MM:SS'] = pd.to_datetime(df['HH:MM:SS'], format='%H:%M:%S')

# Calculate the time difference in seconds and then convert to hours
df['Duration, hour'] = df['HH:MM:SS'].diff().dt.total_seconds() / 3600

df["on-bottom ROP"] = df['Drilling Progression']/df['Duration, hour']


with st.sidebar:
    st.divider()
    st.title("Configure your sections and intervals to splice")
    st.divider()
    # Create a slider for the "Hole Size" column
    selected_hole_size = st.radio("Select Hole Size (inches)", hole_size)

    st.divider()
    # Create a slider for the "Hole Size" column
    selected_bin = st.slider("Select Bin Size (Depth in meters interval)", min_value=10, max_value=200,step=10)
    




# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['WellGUID','WellName','RunId','RunNumber','Hole Size','Depth Interval','TVD', 'FlowRate','OBROP','RPM','WOB','Mud Density','MudType'])



for size in hole_size:
    section_index = df['Hole Diameter'] == size


    # Iterate over unique 'Run' values
    for run_number in df['Run Number'].unique():
        run_index = df['Run Number'] == run_number
        
        # Calculate depth range for each 'Run Number'
        depth_range = df.loc[run_index, 'Hole Depth'].max() - df.loc[run_index, 'Hole Depth'].min()

        # Check if depth range is larger than the selected bin
        if depth_range > selected_bin:
            # Split depth range into intervals using pd.cut
            df.loc[run_index, 'Depth Interval'] = pd.cut(
                df.loc[run_index, 'Hole Depth'],
                bins=range(int(df.loc[run_index, 'Hole Depth'].min()), int(df.loc[run_index, 'Hole Depth'].max()) + selected_bin, selected_bin),
                right=False
            ).astype(str)
        else:
            # Treat the whole depth range as a single interval
            df.loc[run_index, 'Depth Interval'] = f'[{df.loc[run_index, "Hole Depth"].min()}, {df.loc[run_index, "Hole Depth"].max()})'
            
    
    df_bin = df


    # # Use this is range is > bin size
    # df.loc[section_index, 'Depth Interval'] = pd.cut(df.loc[section_index, 'Hole Depth'], bins=range(0, len(df), selected_bin), right=False)

    # df_bin = df

    
    # Calculate average values for each bin interval
    avg_values = df.loc[section_index].groupby('Depth Interval',observed=False).agg({
        'Weight on Bit': 'mean',
        'on-bottom ROP': 'mean',
        'Rotary RPM': 'mean',
        'Total Pump Output' : 'mean'
    }).reset_index()

    # Round the average values to two decimal places
    avg_values = avg_values.round({ 'Weight on Bit': 2, 'on-bottom ROP': 2, 'Rotary RPM': 2, 'Total Pump Output' : 2})

    # Rename columns with 'avg_' prefix
    avg_values.rename(columns={
        'Total Pump Output' : 'FlowRate',
        'on-bottom ROP': 'OBROP',
        'Rotary RPM': 'RPM',
        'Weight on Bit': 'WOB'

    }, inplace=True)

    avg_values["Hole Size"] = size

    # Append the results to the result_df
    result_df = pd.concat([result_df, avg_values], ignore_index=True)


result_df = result_df.dropna(subset=['OBROP'])


df_bin.to_csv(r"C:\Users\ZHamid2\OneDrive - SLB\Downloads\high_freq_bin.csv", index=False)


print(result_df)

col1, col2, col3, col4 = st.columns(4)



# Filter data based on selected hole size
if selected_hole_size != 'All':
    filtered_df = result_df[result_df['Hole Size'] == selected_hole_size]
else:
    filtered_df = result_df

filtered_df['Lower Depth'] = filtered_df['Depth Interval'].str.split(r'\[|,|\)').str[1].astype(float)
lower = filtered_df['Lower Depth']

# Iterate through each column and create individual plots
for column in result_df.columns:
    if (column != 'Depth Interval') and (column != 'Hole Size'):
        def plot(column):
            # Preprocess 'Depth Interval' column to extract lower limit
            # filtered_df['Upper Depth'] = filtered_df['Depth Interval'].str.split('[,)]').str[1].astype(float)
            # filtered_df['Lower Depth'] = filtered_df['Depth Interval'].str.split(r'\[|,|\)').str[1].astype(float)
            
            # Create a scatter plot with x as the column values and y as the lower limit of Depth Interval
            fig, ax = plt.subplots(figsize=(5, 6))
            scatter = ax.scatter(filtered_df[column], filtered_df['Lower Depth'], c=filtered_df[column], cmap='viridis')

            # Invert the y-axis
            ax.invert_yaxis()

            ax.set_title(f'{column} vs. Depth Interval for Hole Size {selected_hole_size}')


            if column == "OBROP":
                column = "OBROP (m/hr)"
            elif column == "WOB":
                column = "WOB (klbf)"
            elif column == "RPM":
                column = "RPM (c/min)"
            elif column == "FlowRate":
                column = "Flowrate (gpm)"

            ax.set_xlabel(column)
            ax.set_ylabel('Depth Interval (m)')
            ax.grid(True)


            # Display the plot in the Streamlit app
            st.plotly_chart(fig,use_container_width=True)

        def plot_all(checkbox):
            # Preprocess 'Depth Interval' column to extract lower limit
            filtered_df['Lower Depth'] = filtered_df['Depth Interval'].str.split('[,)]').str[1].astype(float)

            # Normalize the values of the four parameters between 0 and 1
            scaler = MinMaxScaler()
            normalized_params = scaler.fit_transform(filtered_df[['FlowRate', 'OBROP', 'RPM', 'WOB']])

            # Create a scatter plot with normalized values
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.suptitle('Depth Interval vs. Normalized Parameters')

            if checkbox == "All Parameter Plot":

                ax.scatter(normalized_params[:, 0], filtered_df['Lower Depth'], label='FlowRate (gpm)', cmap='viridis', alpha=0.8)
                ax.scatter(normalized_params[:, 1], filtered_df['Lower Depth'], label='OBROP (m/hr)', cmap='viridis', alpha=0.8)
                ax.scatter(normalized_params[:, 2], filtered_df['Lower Depth'], label='RPM', cmap='viridis', alpha=0.8)
                ax.scatter(normalized_params[:, 3], filtered_df['Lower Depth'], label='WOB (klbf)', cmap='viridis', alpha=0.8)
            
            elif checkbox == "OBROP vs. RPM":

                ax.scatter(normalized_params[:, 1], filtered_df['Lower Depth'], label='OBROP (m/hr)', cmap='viridis', alpha=0.8)
                ax.scatter(normalized_params[:, 2], filtered_df['Lower Depth'], label='RPM', cmap='viridis', alpha=0.8)
            
            elif checkbox == "OBROP vs. WOB":

                ax.scatter(normalized_params[:, 1], filtered_df['Lower Depth'], label='OBROP (m/hr)', cmap='viridis', alpha=0.8)
                ax.scatter(normalized_params[:, 3], filtered_df['Lower Depth'], label='WOB (klbf)', cmap='viridis', alpha=0.8)

            elif checkbox == "Cross Plot":
                # Define minimum and maximum bubble sizes
                min_size = 10
                max_size = 500

                # Create a bubble chart using Plotly
                fig = go.Figure()

                # Normalize OBROP values between 0 and 1 for sizing
                normalized_obrop = (filtered_df['OBROP'] - 102.07) / (303.12 - 102.07)
                normalized_obrop = normalized_obrop.clip(0, 1)  # Clip values to the [0, 1] range

                # Calculate bubble size based on normalized OBROP values
                bubble_size = min_size + (max_size - min_size) * normalized_obrop

                fig.add_trace(go.Scatter(
                    x=filtered_df['WOB'],
                    y=filtered_df['RPM'],
                    mode='markers',
                    marker=dict(
                        size=bubble_size,  # Bubble size based on normalized OBROP values
                        color=filtered_df['OBROP'],  # Color based on OBROP values

                        showscale=True
                    ),
                    text=filtered_df['OBROP'],  # Hover text based on OBROP values
                    hoverinfo='x+y+text'
                ))

                fig.update_layout(
                    title='Cross Plot: RPM vs WOB',
                    xaxis=dict(
                        title='WOB (klbf)'
                    ),
                    yaxis=dict(
                        title='RPM (c/min)',
                    ),
                )



                # Display the Plotly chart in the Streamlit app
                # st.plotly_chart(fig, use_container_width=True)
                
      

            else:
                pass


            ax.set_xlabel('Normalized Parameters')
            ax.set_ylabel('Depth Interval (m)')

            ax.legend(loc="lower right")

            ax.grid(True)


            # Display the plot in the Streamlit app
            st.plotly_chart(fig,use_container_width=True, theme='streamlit')


        if (column == 'RPM'):
            with col1:
                # st.subheader("RPM")
                plot(column)
        elif (column == 'FlowRate'):
            with col2:
                # st.subheader("Flow Rate")
                plot(column)
        elif (column == 'WOB'):
            with col3:
                # st.subheader("WOB")
                plot(column)
        elif (column == 'OBROP'):
            with col4:
                # st.subheader("OBROP")
                plot(column)

st.divider()

st.subheader("Operation Parameter CrossPlots",divider=True)
checkbox = st.selectbox("Select Plots",("All Parameter Plot","OBROP vs. RPM","OBROP vs. WOB", "Cross Plot"))  

col1, col2= st.columns(2)

with col1:
    plot_all(checkbox)
    


        
filtered_df['Lower Depth'] = lower

with col2:
    
    filtered_df = filtered_df.sort_values(by="Lower Depth", ascending=True)
 

    # Rename the column to 'Depth'
    filtered_df.rename(columns={'Depth Interval': 'Depth'}, inplace=True)

    # Specify the path for the output CSV file
    output_csv_path = r"C:\Users\ZHamid2\OneDrive - SLB\Downloads\high_freq_output.csv"
    # Write the DataFrame to a new CSV file
    filtered_df.to_csv(output_csv_path, index=False)

    # Display the DataFrame with highlighting empty cells
    edited_df = st.data_editor(filtered_df,height=500,num_rows="dynamic")
    st.write(f"Current Data Point Rows Count : {len(filtered_df)}")
    st.write(f"Original Data Point Rows Count (Pre-cleaning) : {first_original_rows}")
    st.write(f"Original Data Point Rows Count (Post-cleaning) : {original_rows}")
    st.write(f"Data Points has been trimmed up to : {((first_original_rows - len(filtered_df))/first_original_rows)*100:.2f} %")



    # if download:

    #     # Specify the path for the output CSV file
    #     output_csv_path = r"C:\Users\ZHamid2\OneDrive - SLB\Downloads\high_freq_output.csv"

    #     # Write the DataFrame to a new CSV file
    #     edited_df.to_csv(output_csv_path, index=False)


# Assuming 'Lower Depth' column contains Interval objects or NaN values
filtered_df['Lower Depth'] = filtered_df['Lower Depth'].apply(lambda x: x.right if isinstance(x, pd.Interval) and pd.notnull(x) else x)

st.divider()
st.subheader('Drilling Parameters Correlation Matrix',divider=True)

col4, col5 = st.columns(2)

with col4: 
    # Calculate the correlation matrix
    correlation_matrix = filtered_df[['WOB', 'OBROP', 'RPM', 'FlowRate']].corr()
    
    # Create an interactive heatmap using Plotly Express
    fig = px.imshow(correlation_matrix, labels=dict(color="Correlation Score"), x=correlation_matrix.columns, y=correlation_matrix.columns,color_continuous_scale='Reds')
    fig.update_layout(title='Correlation Heat Map',
                          height =  700)
    
    # Add correlation scores to the scatter matrix
    for i, col in enumerate(['WOB', 'OBROP', 'RPM', 'FlowRate']):
        for j, row in enumerate(['WOB', 'OBROP', 'RPM', 'FlowRate']):
            fig.add_trace(go.Scatter(x=[col], y=[row], text=[f'{correlation_matrix.iloc[i, j]:.2f}'],
                                    mode='text', showlegend=False))

    # Display the interactive heatmap using Streamlit
    st.plotly_chart(fig,use_container_width=True)


with col5: 

        # Assuming filtered_df contains your data
        fig = px.scatter_matrix(filtered_df[['WOB', 'OBROP', 'RPM', 'FlowRate']])
        fig.update_layout(title='Correlation Pair Plot',
                          height =  800)

        # Display the pair plot in Streamlit
        st.plotly_chart(fig,use_container_width=True)


