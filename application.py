# Import necessary libraries
from flask import Flask, render_template, request, redirect
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Define a class to represent the input data
class CustomData:
    def __init__(self,
                 Age: float,
                 Flight_Distance: float,
                 Inflight_wifi_service: float,
                 Departure_Arrival_time_convenient: float,
                 Ease_of_Online_booking: float,
                 Gate_location: float,
                 Food_and_drink: float,
                 Online_boarding: float,
                 Seat_comfort: float,
                 Inflight_entertainment: float,
                 On_board_service: float,
                 Leg_room_service: float,
                 Baggage_handling: float,
                 Checkin_service: float,
                 Inflight_service: float,
                 Cleanliness: float,
                 Arrival_Delay_in_Minutes: float,
                 Gender: float,
                 Customer_Type: float,
                 Type_of_Travel: float,
                 Class: float
                 ):
        # Initialize attributes
        self.Age = Age
        self.Flight_Distance = Flight_Distance
        self.Inflight_wifi_service = Inflight_wifi_service
        self.Departure_Arrival_time_convenient = Departure_Arrival_time_convenient
        self.Ease_of_Online_booking = Ease_of_Online_booking
        self.Gate_location = Gate_location
        self.Food_and_drink = Food_and_drink
        self.Online_boarding = Online_boarding
        self.Seat_comfort = Seat_comfort
        self.Inflight_entertainment = Inflight_entertainment
        self.On_board_service = On_board_service
        self.Leg_room_service = Leg_room_service
        self.Baggage_handling = Baggage_handling
        self.Checkin_service = Checkin_service
        self.Inflight_service = Inflight_service
        self.Cleanliness = Cleanliness
        self.Arrival_Delay_in_Minutes = Arrival_Delay_in_Minutes
        self.Gender = Gender
        self.Customer_Type = Customer_Type
        self.Type_of_Travel = Type_of_Travel
        self.Class = Class

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

# Load the fitted StandardScaler
scaler = joblib.load('scaler.pkl')

# Define a function to make predictions
def make_predictions(scaler, input_data):
    # Check if input_data has the correct format and dimensions
    if not isinstance(input_data, (tuple, list)):
        raise ValueError("Input data must be a tuple or a list.")
    
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    # Make predictions using the model
    predictions = model.predict(std_data)
    
    return predictions

# Define route for the index page
@app.route('/')
def index():
    return redirect('/predict')  # Redirect to the prediction page

# Define route for predicting data
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html', form_data={})  # Render the form template with empty form data
    else:
        # Create CustomData object with form data
        data = CustomData(
            Age=float(request.form.get('Age')),
            Flight_Distance=float(request.form.get('Flight_distance')),
            Inflight_wifi_service=float(request.form.get('Inflight_wifi_service')),
            Departure_Arrival_time_convenient=float(request.form.get('Departure_arrival_time_convenient')),
            Ease_of_Online_booking=float(request.form.get('Ease_of_online_booking')),
            Gate_location=float(request.form.get('Gate_location')),
            Food_and_drink=float(request.form.get('Food_and_drink')),
            Online_boarding=float(request.form.get('Online_boarding')),
            Seat_comfort=float(request.form.get('Seat_comfort')),
            Inflight_entertainment=float(request.form.get('Inflight_entertainment')),
            On_board_service=float(request.form.get('On_board_service')),
            Leg_room_service=float(request.form.get('Leg_room_service')),
            Baggage_handling=float(request.form.get('Baggage_handling')),
            Checkin_service=float(request.form.get('Checkin_service')),
            Inflight_service=float(request.form.get('Inflight_service')),
            Cleanliness=float(request.form.get('Cleanliness')),
            Arrival_Delay_in_Minutes=float(request.form.get('Arrival_delay_in_minutes')),
            Gender=request.form.get('Gender'),
            Customer_Type=request.form.get('Customer_type'),
            Type_of_Travel=request.form.get('Type_of_travel'),
            Class=request.form.get('Class')
        )

        # Perform prediction
        prediction = make_predictions(scaler, [data.Age, data.Flight_Distance, data.Inflight_wifi_service,
                                                data.Departure_Arrival_time_convenient, data.Ease_of_Online_booking,
                                                data.Gate_location, data.Food_and_drink, data.Online_boarding,
                                                data.Seat_comfort, data.Inflight_entertainment, data.On_board_service,
                                                data.Leg_room_service, data.Baggage_handling, data.Checkin_service,
                                                data.Inflight_service, data.Cleanliness, data.Arrival_Delay_in_Minutes,
                                                data.Gender, data.Customer_Type, data.Type_of_Travel, data.Class])

        # Translate prediction to human-readable format
        prediction_text = "The Customer is likely to be neutral or dissatisfied" if prediction[0] == 1 else "The Customer is likely to be satisfied"
        
        # Pass form data back to the template
        form_data = {
            'Age': float(request.form.get('Age')),
            'Flight Distance': float(request.form.get('Flight_distance')),
            'Inflight wifi service': float(request.form.get('Inflight_wifi_service')),
            'Departure/Arrival time convenient': float(request.form.get('Departure_arrival_time_convenient')),
            'Ease of Online booking': float(request.form.get('Ease_of_online_booking')),
            'Gate location': float(request.form.get('Gate_location')),
            'Food and drink': float(request.form.get('Food_and_drink')),
            'Online boarding': float(request.form.get('Online_boarding')),
            'Seat comfort': float(request.form.get('Seat_comfort')),
            'Inflight entertainment': float(request.form.get('Inflight_entertainment')),
            'On-board service': float(request.form.get('On_board_service')),
            'Leg room service': float(request.form.get('Leg_room_service')),
            'Baggage handling': float(request.form.get('Baggage_handling')),
            'Checkin service': float(request.form.get('Checkin_service')),
            'Inflight service': float(request.form.get('Inflight_service')),
            'Cleanliness': float(request.form.get('Cleanliness')),
            'Arrival Delay in Minutes': float(request.form.get('Arrival_delay_in_minutes')),
            'Gender': request.form.get('Gender'),
            'Customer Type': request.form.get('Customer_type'),
            'Type of Travel': request.form.get('Type_of_travel'),
            'Class': request.form.get('Class')
        }

        # Render the template with prediction result and form data
        return render_template('index.html', prediction=prediction_text, form_data=form_data)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
