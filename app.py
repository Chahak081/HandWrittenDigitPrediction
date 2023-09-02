import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from streamlit_drawable_canvas import st_canvas

# Load the dataset
df = load_digits()

# Preprocess the data
data = df.images.reshape(len(df.images), -1) / 16.0

# Load your trained model
rf = RandomForestClassifier()
rf.fit(data, df.target)

# Streamlit app
st.title('Digit Recognition App')

# Create a drawing canvas for the user to draw a digit
st.write('Draw a digit below:')
canvas_result = st_canvas(
    fill_color="black",  # Background color of the canvas
    stroke_width=10,     # Stroke width for drawing
    stroke_color="white",  # Stroke color for drawing
    background_image=None,  # You can provide a background image if needed
    drawing_mode="freedraw",  # Allow free drawing
    key="canvas",
    width=200,            # Width of the canvas
    height=200,           # Height of the canvas
)

# Make predictions when the user clicks a button
if st.button('Predict'):
    if canvas_result.image_data is not None:
        # Resize the drawn image to 8x8
        resized_image = Image.fromarray(canvas_result.image_data).resize((8, 8), Image.ANTIALIAS).convert('L')

        # Convert the image to a NumPy array and flatten it
        resized_image = np.array(resized_image)
        resized_image = resized_image.flatten() / 16.0

        # Make predictions
        prediction = rf.predict([resized_image])

        # Display the predicted digit
        st.write(f'Predicted Digit: {prediction[0]}')


