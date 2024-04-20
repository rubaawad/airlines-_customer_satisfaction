# Import necessary libraries
from flask import Flask, render_template, request, redirect
from sklearn.preprocessing import StandardScaler
# Import pandas library
import pandas as pd
import numpy as np
import joblib
# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

# Load the fitted StandardScaler
scaler = joblib.load('scaler.pkl')

# Define a function to make predictions
def make_predictions(scaler, input_data):
    print ("input data ",input_data)
    
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    # Make predictions using the model
    predictions = model.predict(std_data)
    
    return predictions

# Define route for predicting data
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html', form_data={})  # Render the form template with empty form data
    else:
        # Create CustomData object with form data
        
        Age=float(request.form.get('Age'))
        Flight_Distance=float(request.form.get('Flight_distance'))
        Inflight_wifi_service=float(request.form.get('Inflight_wifi_service'))
        Departure_Arrival_time_convenient=float(request.form.get('Departure_arrival_time_convenient'))
        Ease_of_Online_booking=float(request.form.get('Ease_of_online_booking'))
        Gate_location=float(request.form.get('Gate_location'))
        Food_and_drink=float(request.form.get('Food_and_drink'))
        Online_boarding=float(request.form.get('Online_boarding'))
        Seat_comfort=float(request.form.get('Seat_comfort'))
        Inflight_entertainment=float(request.form.get('Inflight_entertainment'))
        On_board_service=float(request.form.get('On_board_service'))
        Leg_room_service=float(request.form.get('Leg_room_service'))
        Baggage_handling=float(request.form.get('Baggage_handling'))
        Checkin_service=float(request.form.get('Checkin_service'))
        Inflight_service=float(request.form.get('Inflight_service'))
        Cleanliness=float(request.form.get('Cleanliness'))
        Arrival_Delay_in_Minutes=float(request.form.get('Arrival_delay_in_minutes'))
        Gender=request.form.get('Gender')
        Customer_Type=request.form.get('Customer_type')
        Type_of_Travel=request.form.get('Type_of_travel')
        Class=request.form.get('Class')
        
        data = (Gender,Customer_Type,Age,Type_of_Travel,Class,Flight_Distance,Inflight_wifi_service,Departure_Arrival_time_convenient,Ease_of_Online_booking,Gate_location,Food_and_drink,Online_boarding,Seat_comfort,
                Inflight_entertainment,On_board_service,Leg_room_service,Baggage_handling,Checkin_service,Inflight_service,Cleanliness,Arrival_Delay_in_Minutes)

        # Perform prediction
        prediction = make_predictions(scaler, data)
        print("prediction",prediction[0])

        # Translate prediction to human-readable format
        prediction_text = "Happy" if prediction[0] == 1 else "Sad"
        
        # Pass form data back to the template
        form_data = {
            'Age': float(request.form.get('Age')),
            'Flight_distance': float(request.form.get('Flight_distance')),
            'Inflight_wifi_service': float(request.form.get('Inflight_wifi_service')),
            'Departure_arrival_time_convenient': float(request.form.get('Departure_arrival_time_convenient')),
            'Ease_of_online_booking': float(request.form.get('Ease_of_online_booking')),
            'Gate_location': float(request.form.get('Gate_location')),
            'Food_and_drink': float(request.form.get('Food_and_drink')),
            'Online_boarding': float(request.form.get('Online_boarding')),
            'Seat_comfort': float(request.form.get('Seat_comfort')),
            'Inflight_entertainment': float(request.form.get('Inflight_entertainment')),
            'On_board_service': float(request.form.get('On_board_service')),
            'Leg_room_service': float(request.form.get('Leg_room_service')),
            'Baggage_handling': float(request.form.get('Baggage_handling')),
            'Checkin_service': float(request.form.get('Checkin_service')),
            'Inflight_service': float(request.form.get('Inflight_service')),
            'Cleanliness': float(request.form.get('Cleanliness')),
            'Arrival_delay_in_minutes': float(request.form.get('Arrival_delay_in_minutes')),
            'Gender': request.form.get('Gender'),
            'Customer_type': request.form.get('Customer_type'),
            'Type_of_travel': request.form.get('Type_of_travel'),
            'Class': request.form.get('Class')
        }

        # Render the template with prediction result and form data
        return render_template('index.html', prediction=prediction_text, form_data=form_data)
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
