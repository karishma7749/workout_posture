import cv2
import numpy as np

def create_panel_background(width, height, color=(18, 18, 24)):
    """Create a modern-looking panel background"""
    panel = np.ones((height, width, 3), dtype="uint8") * color
    return panel

def add_modern_title(panel, title, y_pos=40):
    """Add modern title styling"""
    # Add sleek title bar
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 70), (25, 25, 35), -1)
    
    # Add accent line
    accent_color = (0, 200, 255)  # Modern blue accent
    cv2.line(panel, (20, y_pos + 15), (panel.shape[1] - 20, y_pos + 15), accent_color, 2)
    
    # Add title text with modern font
    cv2.putText(panel, title, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.9, (230, 230, 230), 1)

def add_modern_info_box(panel, info_dict, start_y):
    """Create a modern info box with exercise analysis"""
    box_height = 150
    box_width = panel.shape[1] - 40
    
    # Create info box background
    cv2.rectangle(panel, (20, start_y), (20 + box_width, start_y + box_height), (25, 25, 35), -1)
    cv2.rectangle(panel, (20, start_y), (20 + box_width, start_y + box_height), (30, 30, 45), 1)
    
    # Title section
    cv2.rectangle(panel, (20, start_y), (20 + box_width, start_y + 30), (30, 30, 45), -1)
    cv2.putText(panel, "Exercise Analysis", (35, start_y + 20), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)
    
    # Main info
    x_pos = 35
    y_offset = start_y + 60
    line_spacing = 30

    labels = ["Exercise:", "Confidence:", "Form Quality:"]
    values = [info_dict['Exercise'], 
              info_dict['Confidence'],
              info_dict['Form Quality']]

    for label_text, value in zip(labels, values):
        cv2.putText(panel, label_text, (x_pos, y_offset), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 1)
        cv2.putText(panel, str(value), (x_pos + 150, y_offset), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (230, 230, 230), 1)
        y_offset += line_spacing

def add_confidence_display(panel, predictions, label, start_y):
    """Display confidence levels for each pose in a separate window"""
    box_height = 250
    box_width = panel.shape[1] - 40
    
    # Create box background
    cv2.rectangle(panel, (20, start_y), (20 + box_width, start_y + box_height), (25, 25, 35), -1)
    cv2.rectangle(panel, (20, start_y), (20 + box_width, start_y + box_height), (30, 30, 45), 1)
    
    # Title section
    cv2.rectangle(panel, (20, start_y), (20 + box_width, start_y + 30), (30, 30, 45), -1)
    cv2.putText(panel, "Pose Confidence Levels", (35, start_y + 20), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)
    
    # Display pose confidences in two columns
    x_pos_left = 35
    x_pos_right = box_width // 2 + 20
    y_offset = start_y + 60
    line_spacing = 25
    
    num_poses = min(len(predictions), len(label))  # Ensure we don't exceed available labels
    for i in range(num_poses):
        pose_text = label[i].replace('_', ' ').title()  # Format pose name
        confidence = predictions[i]
        pose_display = f"{pose_text}: {confidence:.1%}"
        
        if i % 2 == 0:  # Even index, display in left column
            x_pos = x_pos_left
        else:  # Odd index, display in right column
            x_pos = x_pos_right
        
        cv2.putText(panel, pose_display, (x_pos, y_offset), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 180), 1)
        
        if i % 2 == 1:  # Increment y-offset after every two entries
            y_offset += line_spacing

def add_footer(main_window, window_height, window_width):
    """Add modern footer to the main window"""
    footer_height = 40
    cv2.rectangle(main_window, (0, window_height-footer_height), 
                 (window_width, window_height), (25, 25, 35), -1)
    cv2.putText(main_window, "ExerciseAI Pro", (20, window_height-15), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 1)
    cv2.putText(main_window, "ESC to exit", (window_width-200, window_height-15), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 1)

def add_panel_borders(panel, margin_x, margin_y, display_width, display_height):
    """Add borders to a panel"""
    cv2.rectangle(panel, (margin_x-2, margin_y-2), 
                 (margin_x+display_width+2, margin_y+display_height+2), 
                 (30, 30, 45), 2)

def add_panel_separator(main_window, x, window_height):
    """Add a subtle separator between panels"""
    cv2.line(main_window, (x, 0), (x, window_height), (30, 30, 45), 2) 