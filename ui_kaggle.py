import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
from PIL import Image
import io
import rembg

# Set page configuration
st.set_page_config(
    page_title="Leaf Disease Severity Analysis",
    page_icon="🍃",
    layout="wide"
)

# Title and description
st.title("Leaf Disease Severity Analysis")
st.markdown("""
This application analyzes leaf images to detect and quantify disease severity.
Upload an image of a plant leaf to get started.
""")

# Define the image processing functions
def preprocess_image(image):
    # Convert PIL image to OpenCV format
    image = np.array(image.convert('RGB'))
    
    # Create a copy for visualization
    original = image.copy()
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    return original, blurred, image

def segment_leaf(original_image, pil_image):
    """
    Segment the leaf using either background removal or color thresholding
    """
    # Try background removal with rembg
    try:
        with st.spinner("Removing background..."):
            # Process the image with rembg to remove background
            output = rembg.remove(pil_image)
            
            # Convert to numpy array
            output_array = np.array(output)
            
            # Extract the alpha channel as the mask
            if output_array.shape[2] == 4:  # RGBA
                # The alpha channel is our mask
                leaf_mask = output_array[:, :, 3]
            else:
                raise ValueError("No alpha channel found")
                
            # Ensure mask is binary and properly formatted
            _, leaf_mask = cv2.threshold(leaf_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Clean up the mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
            leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate total leaf area (pixel count)
            total_leaf_pixels = np.sum(leaf_mask > 0)
            
            # If we got a good mask, use it
            if total_leaf_pixels > 1000:  # arbitrary threshold to ensure we got something useful
                leaf_mask = leaf_mask.astype(np.uint8)
                segmented_leaf = cv2.bitwise_and(original_image, original_image, mask=leaf_mask)
                return segmented_leaf, leaf_mask, total_leaf_pixels
            else:
                raise ValueError("Mask too small or empty")
    
    except Exception as e:
        # If rembg fails, fall back to color thresholding
        st.warning(f"Background removal failed: {str(e)}. Using color thresholding instead.")
        
        # Convert to HSV
        hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
        
        # Define range for green color - adjust if needed for your leaves
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create mask for green parts
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, 0
        
        # Find the largest contour (assumed to be the leaf)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask with just the largest contour
        leaf_mask = np.zeros_like(mask)
        cv2.drawContours(leaf_mask, [max_contour], 0, 255, -1)
        
        # Apply the mask to get the segmented leaf
        segmented_leaf = cv2.bitwise_and(original_image, original_image, mask=leaf_mask)
        
        # Calculate total leaf area (pixel count)
        total_leaf_pixels = np.sum(leaf_mask > 0)
        
        return segmented_leaf, leaf_mask, total_leaf_pixels

def segment_disease(segmented_leaf, leaf_mask):
    """
    Segment diseased regions using simple but effective thresholding
    """
    # Convert leaf to HSV
    hsv_leaf = cv2.cvtColor(segmented_leaf, cv2.COLOR_RGB2HSV)
    
    # Create masks for different disease symptoms
    
    # 1. Yellowish/brownish regions (common disease symptom)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv_leaf, lower_yellow, upper_yellow)
    
    # 2. Dark spots (necrosis)
    gray_leaf = cv2.cvtColor(segmented_leaf, cv2.COLOR_RGB2GRAY)
    _, dark_spots = cv2.threshold(gray_leaf, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Ensure we only look at dark spots within the leaf
    dark_spots = cv2.bitwise_and(dark_spots, leaf_mask)
    
    # Combine the masks
    disease_mask = cv2.bitwise_or(yellow_mask, dark_spots)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    min_size = 50  # Minimum size of spots to consider
    labeled_mask = label(disease_mask)
    disease_mask_clean = remove_small_objects(labeled_mask, min_size=min_size)
    disease_mask_clean = disease_mask_clean > 0
    disease_mask_clean = disease_mask_clean.astype(np.uint8) * 255
    
    # Special case for predominantly yellow leaves
    hsv_mean = cv2.mean(hsv_leaf, mask=leaf_mask)
    
    # If the leaf is mostly yellow (potential systemic disease)
    if 15 <= hsv_mean[0] <= 35 and hsv_mean[1] >= 30:
        yellow_pixels = np.sum(yellow_mask > 0)
        leaf_pixels = np.sum(leaf_mask > 0)
        yellow_percentage = (yellow_pixels / leaf_pixels) * 100
        
        # If over 70% of the leaf is yellow, consider the entire leaf diseased
        if yellow_percentage > 70:
            disease_mask_clean = leaf_mask.copy()
    
    # Visualize the disease regions on the original leaf
    disease_regions = cv2.bitwise_and(segmented_leaf, segmented_leaf, mask=disease_mask_clean)
    
    # Count disease pixels
    disease_pixels = np.sum(disease_mask_clean > 0)
    
    return disease_regions, disease_mask_clean, disease_pixels

def calculate_severity(disease_pixels, total_leaf_pixels, segmented_leaf, leaf_mask):
    """
    Calculate disease severity percentage and classify
    """
    if total_leaf_pixels == 0:
        return 0, "No leaf detected", "gray"
    
    # Calculate basic severity percentage
    severity_percentage = (disease_pixels / total_leaf_pixels) * 100
    
    # Check leaf color
    hsv_leaf = cv2.cvtColor(segmented_leaf, cv2.COLOR_RGB2HSV)
    hsv_mean = cv2.mean(hsv_leaf, mask=leaf_mask)
    
    # Simple classification with adjustments for leaf color
    if severity_percentage < 5:
        severity_class = "Healthy (0-5%)"
        severity_color = "green"
    elif severity_percentage < 15:
        severity_class = "Low Infection (5-15%)"
        severity_color = "yellowgreen"
    elif severity_percentage < 30:
        severity_class = "Moderate Infection (15-30%)"
        severity_color = "orange"
    elif severity_percentage < 50:
        severity_class = "High Infection (30-50%)"
        severity_color = "orangered"
    else:
        severity_class = "Severe Infection (>50%)"
        severity_color = "red"
    
    # Adjustment for yellow leaves
    # If mean hue is in yellow range (15-35)
    if 15 <= hsv_mean[0] <= 35 and hsv_mean[1] >= 30 and severity_percentage < 15:
        # Increase minimum severity for yellow leaves
        severity_percentage = max(severity_percentage, 15)
        severity_class = "Yellow Leaf - Moderate Infection"
        severity_color = "orange"
    
    return severity_percentage, severity_class, severity_color

def plot_results(original, segmented_leaf, disease_regions, leaf_mask, disease_mask, 
                severity_percentage, severity_class):
    """
    Create visualization of results
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3)
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Segmented leaf
    axes[0, 1].imshow(segmented_leaf)
    axes[0, 1].set_title('Segmented Leaf')
    axes[0, 1].axis('off')
    
    # Disease regions
    axes[0, 2].imshow(disease_regions)
    axes[0, 2].set_title('Disease Regions')
    axes[0, 2].axis('off')
    
    # Leaf mask
    axes[1, 0].imshow(leaf_mask, cmap='gray')
    axes[1, 0].set_title('Leaf Mask')
    axes[1, 0].axis('off')
    
    # Disease mask
    axes[1, 1].imshow(disease_mask, cmap='gray')
    axes[1, 1].set_title('Disease Mask')
    axes[1, 1].axis('off')
    
    # Pie chart for severity
    labels = ['Healthy', 'Diseased']
    sizes = [100 - severity_percentage, severity_percentage]
    colors = ['green', 'red']
    explode = (0, 0.1)
    
    axes[1, 2].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    axes[1, 2].set_title(f'Severity: {severity_percentage:.2f}%\n{severity_class}')
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def analyze_leaf_disease(image):
    """
    Main function to analyze leaf disease severity
    """
    with st.spinner("Processing image..."):
        # Step 1: Preprocessing
        original, hsv_blurred, rgb_image = preprocess_image(image)
        
        # Step 2: Leaf segmentation
        segmented_leaf, leaf_mask, total_leaf_pixels = segment_leaf(rgb_image, image)
        
        if segmented_leaf is None:
            st.error("No leaf detected in the image. Please try another image.")
            return None
        
        # Step 3: Disease segmentation
        disease_regions, disease_mask, disease_pixels = segment_disease(segmented_leaf, leaf_mask)
        
        # Step 4: Calculate severity
        severity_percentage, severity_class, severity_color = calculate_severity(
            disease_pixels, total_leaf_pixels, segmented_leaf, leaf_mask)
        
        # Step 5: Generate results plot
        results_plot = plot_results(original, segmented_leaf, disease_regions, leaf_mask, 
                                  disease_mask, severity_percentage, severity_class)
        
        return {
            'total_leaf_pixels': total_leaf_pixels,
            'disease_pixels': disease_pixels,
            'severity_percentage': severity_percentage,
            'severity_class': severity_class,
            'severity_color': severity_color,
            'results_plot': results_plot
        }

# Create the Streamlit interface
# Sidebar for additional options
with st.sidebar:
    st.header("Settings")
    st.markdown("---")
    
    st.markdown("### About")
    st.info("""
    This tool analyzes leaf images to detect disease severity by:
    1. Segmenting the leaf from the background
    2. Identifying diseased regions on the leaf
    3. Calculating the severity percentage
    4. Classifying the disease severity level
    """)
    
    st.markdown("### Disclaimer")
    st.warning("""
    This is a prototype tool and should not replace professional 
    plant pathology diagnosis. Results may vary depending on 
    image quality and disease type.
    """)

# Main content area
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Analyze the image and display results
    results = analyze_leaf_disease(image)
    
    if results:
        with col2:
            # Display the severity with colored box
            st.markdown(f"""
            <div style="background-color: {results['severity_color']}; 
                        padding: 20px; 
                        border-radius: 10px; 
                        text-align: center;
                        color: white;
                        font-weight: bold;
                        margin-bottom: 20px;">
                <h2>Disease Severity: {results['severity_percentage']:.2f}%</h2>
                <h3>{results['severity_class']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display detailed metrics
            st.markdown("### Detailed Metrics")
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Total Leaf Area", f"{results['total_leaf_pixels']} pixels")
            with metrics_col2:
                st.metric("Diseased Area", f"{results['disease_pixels']} pixels")
        
        # Display the analysis visualization
        st.image(Image.open(results['results_plot']), caption="Analysis Results", use_column_width=True)
        
        # Add explanation of the analysis process
        with st.expander("View Analysis Process"):
            st.markdown("""
            ### Analysis Steps:
            
            1. **Image Preprocessing**: The image is prepared for processing.
            
            2. **Leaf Segmentation**: The leaf is separated from the background using one of two methods:
               - Background removal with rembg (when possible)
               - Color-based segmentation (as fallback)
            
            3. **Disease Detection**: The system detects:
               - Yellow/brown spots (common disease symptoms)
               - Dark spots (indicating necrosis or rot)
               - Overall yellowing (systemic infection)
            
            4. **Severity Calculation**: 
               - The ratio of diseased pixels to total leaf pixels
               - Adjustment for predominantly yellow leaves
            
            5. **Classification**: The severity percentage is used to classify the infection level
            """)
else:
    # Display instructions when no image is uploaded
    st.markdown("""
    ### Instructions:
    1. Use the file uploader above to select a leaf image
    2. Make sure the leaf is clearly visible with good lighting
    3. The system works best with images where:
       - The disease symptoms are visible
       - The image is well-lit
       - The leaf is the main subject of the image
    """)
    
    # Display sample image placeholders
    st.markdown("### Sample Results Will Appear Here")
    placeholder_cols = st.columns(3)
    with placeholder_cols[0]:
        st.markdown("Original Image")
        st.image("https://via.placeholder.com/300x200?text=Original+Image", use_column_width=True)
    with placeholder_cols[1]:
        st.markdown("Segmented Leaf")
        st.image("https://via.placeholder.com/300x200?text=Segmented+Leaf", use_column_width=True)
    with placeholder_cols[2]:
        st.markdown("Disease Detection")
        st.image("https://via.placeholder.com/300x200?text=Disease+Detection", use_column_width=True)

# Add footer
st.markdown("---")
st.markdown("Leaf Disease Severity Analysis Tool | Developed with Streamlit, OpenCV and scikit-image")