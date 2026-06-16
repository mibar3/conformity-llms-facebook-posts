import os 
import base64
from PIL import Image
import io
import webbrowser
from itertools import product
from IPython.display import HTML, display

def generate_facebook_post(profile_name, post_text, post_time, like_count, dislike_count, 
                          profile_image_path=None, post_image_path=None, output_file=None):
    """
    Generate an HTML file mimicking a Facebook post with custom content and images.
    
    Args:
        profile_name (str): Name of the profile posting
        post_text (str): Text content of the post
        post_time (str): Time of posting (e.g., "Yesterday at 2:00 AM")
        like_count (int): Number of likes
        dislike_count (int): Number of dislikes
        profile_image_path (str, optional): Path to profile image
        post_image_path (str, optional): Path to post image
        output_file (str, optional): Output HTML file name (None for no file output)
    
    Returns:
        str: HTML content as a string
    """
    # Convert images to base64 if provided
    profile_image_base64 = ""
    if profile_image_path and os.path.exists(profile_image_path):
        with open(profile_image_path, "rb") as img_file:
            profile_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
    post_image_base64 = ""
    if post_image_path and os.path.exists(post_image_path):
        with open(post_image_path, "rb") as img_file:
            post_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Determine image formats
    profile_image_format = "jpeg"
    if profile_image_path:
        profile_image_format = profile_image_path.split('.')[-1].lower()
        if profile_image_format not in ['jpeg', 'jpg', 'png', 'gif']:
            profile_image_format = "jpeg"
    
    post_image_format = "jpeg"
    if post_image_path:
        post_image_format = post_image_path.split('.')[-1].lower()
        if post_image_format not in ['jpeg', 'jpg', 'png', 'gif']:
            post_image_format = "jpeg"
    
    # HTML template
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facebook Post Template</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        
        body {{
            background-color: #f0f2f5;
            padding: 20px;
            display: flex;
            justify-content: center;
        }}
        
        .post-container {{
            width: 100%;
            max-width: 680px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .post-header {{
            display: flex;
            padding: 12px 16px;
            align-items: center;
            position: relative;
        }}
        
        .profile-img {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 8px;
            background-color: #e4e6eb;
            overflow: hidden;
        }}
        
        .profile-img img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        
        .post-info {{
            flex: 1;
        }}
        
        .profile-name {{
            font-weight: 600;
            font-size: 15px;
            color: #050505;
            text-decoration: none;
            display: flex;
            align-items: center;
        }}
        
        .profile-name:hover {{
            text-decoration: underline;
        }}
        
        .follow-btn {{
            color: #1877f2;
            font-weight: 600;
            margin-left: 8px;
            font-size: 14px;
            text-decoration: none;
        }}
        
        .post-time {{
            color: #65676b;
            font-size: 13px;
            display: flex;
            align-items: center;
        }}
        
        .privacy-icon {{
            margin-left: 4px;
            font-size: 12px;
        }}
        
        .post-options {{
            color: #65676b;
            font-size: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
        }}
        
        .post-close {{
            margin-left: 8px;
            cursor: pointer;
            color: #65676b;
        }}
        
        .post-content {{
            padding: 4px 16px 16px;
            font-size: 15px;
            color: #050505;
        }}
        
        .post-image {{
            width: 100%;
            max-height: 800px;
            overflow: hidden;
        }}
        
        .post-image img {{
            width: 80%;
            object-fit: contain;
        }}
        
        .engagement {{
            padding: 10px 16px;
            border-bottom: 1px solid #e4e6eb;
        }}
        
        .reactions {{
            display: flex;
            align-items: center;
        }}
        
        .reaction-counts {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        
        .reaction-item {{
            display: flex;
            align-items: center;
            font-size: 16px;
        }}
        
        .reaction-icon {{
            margin-right: 6px;
            font-size: 22px;
        }}
        
        .reaction-count {{
            color: #65676b;
            font-size: 16px;
            font-weight: 500;
        }}
        
        .actions {{
            display: flex;
            justify-content: space-around;
            padding: 4px 0;
            border-bottom: 1px solid #e4e6eb;
        }}
        
        .action-btn {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px 0;
            color: #65676b;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            border-radius: 4px;
        }}
        
        .action-btn:hover {{
            background-color: #f0f2f5;
        }}
        
        .action-icon {{
            margin-right: 8px;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="post-container">
        <div class="post-header">
            <div class="profile-img">
                {"<img src='data:image/" + profile_image_format + ";base64," + profile_image_base64 + "' alt='Profile'>" if profile_image_base64 else "<img src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDQwIDQwIj48cmVjdCB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIGZpbGw9IiNlNGU2ZWIiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIyMHB4IiBmaWxsPSIjODg4Ij48L3RleHQ+PC9zdmc+' alt='Profile'>"}
            </div>
            <div class="post-info">
                <div>
                    <a href="#" class="profile-name">{profile_name}</a>
                    <a href="#" class="follow-btn">Follow</a>
                </div>
                <div class="post-time">
                    <span>{post_time}</span>
                    <span class="privacy-icon">· ⚫</span>
                </div>
            </div>
            <div class="post-options">
                <span>...</span>
                <span class="post-close">×</span>
            </div>
        </div>
        
        <div class="post-content">
            {post_text}
        </div>
        
        <div class="post-image">
            {"<img src='data:image/" + post_image_format + ";base64," + post_image_base64 + "' alt='Post image'>" if post_image_base64 else ""}
        </div>
        
        <div class="engagement">
            <div class="reactions">
                <div class="reaction-counts">
                    <div class="reaction-item">
                        <span class="reaction-icon">👍</span>
                        <span class="reaction-count">{like_count}</span>
                    </div>
                    <div class="reaction-item">
                        <span class="reaction-icon">👎</span>
                        <span class="reaction-count">{dislike_count}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="actions">
            <div class="action-btn" onclick="updateLikes()">
                <span class="action-icon">👍</span>
                <span>Like</span>
            </div>
            <div class="action-btn" onclick="updateDislikes()">
                <span class="action-icon">👎</span>
                <span>Dislike</span>
            </div>
            <div class="action-btn">
                <span class="action-icon">↗️</span>
                <span>Share</span>
            </div>
        </div>
    </div>
    
    <script>
        // Variable to track current reaction counts
        let currentLikes = {like_count};
        let currentDislikes = {dislike_count};
        
        // Function to simulate like button interaction
        function updateLikes() {{
            currentLikes++;
            document.querySelectorAll('.reaction-count')[0].textContent = currentLikes;
        }}
        
        // Function to simulate dislike button interaction
        function updateDislikes() {{
            currentDislikes++;
            document.querySelectorAll('.reaction-count')[1].textContent = currentDislikes;
        }}
    </script>
</body>
</html>
    """
    
    # Write the HTML file if output_file is provided
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"Facebook post generated successfully: {os.path.abspath(output_file)}")
    
    return html_template
