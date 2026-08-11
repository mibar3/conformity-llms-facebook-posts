import os
import base64
from PIL import Image
import io
import webbrowser
from IPython.display import HTML, display

REACTION_DEFS = {
    "like":  ("👍", "#1877f2"),
    "love":  ("❤️", "#f33e58"),
    "haha":  ("😆", "#f7b928"),
    "wow":   ("😮", "#f7b928"),
    "sad":   ("😢", "#f7b928"),
    "angry": ("😡", "#e9710f"),
}

def format_count(n):
    """Format large numbers like Facebook does (1.2K, 3.4M, etc.)"""
    if n == 0:
        return None
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M".rstrip('0').rstrip('.')
    if n >= 1_000:
        return f"{n/1_000:.1f}K".rstrip('0').rstrip('.')
    return str(n)

def build_reaction_bubbles_html(reactions: dict) -> str:
    active = sorted([(k, v) for k, v in reactions.items() if v > 0],
                    key=lambda x: x[1], reverse=True)
    if not active:
        return ""
    
    bubbles = ""
    for name, count in active:
        emoji, color = REACTION_DEFS[name]
        formatted = format_count(count)
        bubbles += (
            f'<span class="reaction-bubble">{emoji}</span>'
            f'<span class="reaction-count">&nbsp;{formatted}</span>'
            f'&nbsp;&nbsp;'
        )
    
    return f'<div class="reaction-item"><div class="reaction-icons">{bubbles}</div></div>'

def generate_facebook_post(profile_name, post_text, post_time,
                            reactions=None, comment_count=0, share_count=0,
                            profile_image_path=None, post_image_path=None,
                            output_file=None, verified=False):
    if reactions is None:
        reactions = {}
    # Convert images to base64 if provided
    profile_image_base64 = ""
    if profile_image_path and os.path.exists(profile_image_path):
        with open(profile_image_path, "rb") as img_file:
            profile_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    post_image_base64 = ""
    if post_image_path and os.path.exists(post_image_path):
        with open(post_image_path, "rb") as img_file:
            post_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

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

    reaction_bubbles_html = build_reaction_bubbles_html(reactions)

    formatted_comments = format_count(comment_count)
    formatted_shares   = format_count(share_count)
    
    right_parts = []
    if formatted_comments:
        right_parts.append(f'{formatted_comments} Comments')
    if formatted_shares:
        right_parts.append(f'{formatted_shares} Shares')
    engagement_right_html = ' · '.join(right_parts)

    # Verified badge SVG (Facebook blue checkmark)
    verified_badge = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16"
             style="margin-left:4px; vertical-align:middle; flex-shrink:0;">
          <circle cx="12" cy="12" r="12" fill="#1877f2"/>
          <path d="M9.5 16.5l-4-4 1.41-1.41L9.5 13.67l7.59-7.59L18.5 7.5z" fill="white"/>
        </svg>
    """ if verified else ""

    # Globe SVG icon (Facebook-style public privacy indicator)
    globe_icon = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="12" height="12"
             style="margin-left:4px; vertical-align:middle; fill:#65676b;">
          <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm0 1.5a6.5 6.5 0 0 1 4.87 10.79c-.34-.69-1.05-1.29-2.37-1.79-.49-.19-1-.35-1.5-.48V8.5h1.5a.5.5 0 0 0 0-1H9V6.5A1.5 1.5 0 0 1 10.5 5h.25a.5.5 0 0 0 0-1H10.5A2.5 2.5 0 0 0 8 6.5V7.5H6.5a.5.5 0 0 0 0 1H8v1.48c-.51.13-1.02.29-1.5.48-1.32.5-2.03 1.1-2.37 1.79A6.5 6.5 0 0 1 8 1.5z"/>
        </svg>
    """

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facebook Post</title>
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
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        /* ── Header ── */
        .post-header {{
            display: flex;
            padding: 12px 16px;
            align-items: center;
        }}

        .profile-img {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 8px;
            background-color: #e4e6eb;
            overflow: hidden;
            flex-shrink: 0;
        }}

        .profile-img img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        .post-info {{ flex: 1; }}

        .name-row {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 4px;
        }}

        .profile-name {{
            font-weight: 600;
            font-size: 15px;
            color: #050505;
            text-decoration: none;
        }}

        .profile-name:hover {{ text-decoration: underline; }}

        .follow-btn {{
            color: #1877f2;
            font-weight: 600;
            font-size: 14px;
            text-decoration: none;
            margin-left: 4px;
        }}

        .post-meta {{
            display: flex;
            align-items: center;
            color: #65676b;
            font-size: 13px;
            margin-top: 2px;
            gap: 2px;
        }}

        .post-options {{
            display: flex;
            align-items: center;
            gap: 4px;
            color: #65676b;
            font-size: 22px;
            cursor: pointer;
            padding: 4px;
        }}

        /* ── Post text ── */
        .post-content {{
            padding: 4px 16px 12px;
            font-size: 15px;
            color: #050505;
            line-height: 1.5;
            white-space: pre-wrap;
        }}

        /* ── Post image ── */
        .post-image {{
            width: 100%;
            line-height: 0;
        }}

        .post-image img {{
            width: 100%;
            object-fit: cover;
            display: block;
        }}

        /* ── Engagement row ── */
        .engagement {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 16px;
            border-bottom: 1px solid #e4e6eb;
            min-height: 36px;
        }}

        .reaction-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .reaction-icons {{
            display: flex;
            gap: 4px;
        }}

        .reaction-bubble {{
        width: 22px; height: 22px;
        border-radius: 50%;
        border: 2px solid white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        position: relative;
    }}

        .reaction-count {{
            color: #65676b;
            font-size: 15px;
        }}

        .reaction-count:hover {{
            text-decoration: underline;
            cursor: pointer;
        }}

        .comment-share-counts {{
            color: #65676b;
            font-size: 15px;
            cursor: pointer;
        }}

        .comment-share-counts:hover {{ text-decoration: underline; }}

        /* ── Action buttons ── */
        .actions {{
            display: flex;
            justify-content: space-around;
            padding: 2px 8px;
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
            gap: 6px;
            user-select: none;
        }}

        .action-btn:hover {{ background-color: #f0f2f5; }}

        .action-btn.liked {{
            color: #1877f2;
        }}

        .action-icon {{ font-size: 20px; }}
    </style>
</head>
<body>
<div class="post-container">

    <!-- Header -->
    <div class="post-header">
        <div class="profile-img">
            {"<img src='data:image/" + profile_image_format + ";base64," + profile_image_base64 + "' alt='Profile'>" if profile_image_base64 else
             "<img src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDQwIDQwIj48cmVjdCB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIGZpbGw9IiNlNGU2ZWIiLz48L3N2Zz4=' alt='Profile'>"}
        </div>
        <div class="post-info">
            <div class="name-row">
                <a href="#" class="profile-name">{profile_name}</a>
                {verified_badge}
                <a href="#" class="follow-btn">· Follow</a>
            </div>
            <div class="post-meta">
                <span>{post_time}</span>
                <span>·</span>
                {globe_icon}
            </div>
        </div>
        <div class="post-options">
            <svg viewBox="0 0 20 20" width="20" height="20" fill="#65676b">
                <circle cx="4" cy="10" r="2"/><circle cx="10" cy="10" r="2"/><circle cx="16" cy="10" r="2"/>
            </svg>
            <svg viewBox="0 0 20 20" width="16" height="16" fill="#65676b">
                <path d="M4 4l12 12M16 4L4 16" stroke="#65676b" stroke-width="2" stroke-linecap="round"/>
            </svg>
        </div>
    </div>

    <!-- Post text -->
    <div class="post-content">{post_text}</div>

    <!-- Post image -->
    <div class="post-image">
        {"<img src='data:image/" + post_image_format + ";base64," + post_image_base64 + "' alt='Post image'>" if post_image_base64 else ""}
    </div>

    <!-- Engagement counts -->
    <div class="engagement">
        <div>{reaction_bubbles_html}</div>
        <div class="engagement-right">{engagement_right_html}</div>
    </div>
    <!-- Action buttons -->
    <div class="actions">
        <div class="action-btn" id="likeBtn" onclick="toggleLike()">
            <span class="action-icon">👍</span>
            <span id="likeBtnLabel">Like</span>
        </div>
        <div class="action-btn" onclick="focusComment()">
            <span class="action-icon">💬</span>
            <span>Comment</span>
        </div>
        <div class="action-btn">
            <span class="action-icon">↗️</span>
            <span>Share</span>
        </div>
    </div>

</div>

<script>
    let liked = false;
    let currentLikes = {reactions.get('like', 0)};

    function toggleLike() {{
        liked = !liked;
        const btn = document.getElementById('likeBtn');
        const label = document.getElementById('likeBtnLabel');
        const countEl = document.querySelector('.reaction-count');

        if (liked) {{
            btn.classList.add('liked');
            label.textContent = 'Liked';
            currentLikes++;
        }} else {{
            btn.classList.remove('liked');
            label.textContent = 'Like';
            currentLikes--;
        }}

        // Update or create the like count display
        if (countEl) {{
            countEl.textContent = formatCount(currentLikes);
        }} else if (liked) {{
            // If there was no like count before, inject the element
            const engLeft = document.querySelector('.engagement > div:first-child');
            engLeft.innerHTML = `
                <div class="reaction-item">
                    <span class="reaction-icons">
                        <span class="reaction-bubble like-bubble">👍</span>
                    </span>
                    <span class="reaction-count">1</span>
                </div>`;
        }}
    }}

    function formatCount(n) {{
        if (n <= 0) return null;
        if (n >= 1000000) return (n / 1000000).toFixed(1).replace(/\\.0$/, '') + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1).replace(/\\.0$/, '') + 'K';
        return n.toString();
    }}

    function focusComment() {{
        // Scrolls to bottom to simulate opening comment box
        window.scrollTo({{top: document.body.scrollHeight, behavior: 'smooth'}});
    }}
</script>
</body>
</html>
"""

    if output_file:
        # Create the parent directory if it doesn't exist yet -- most callers write into a
        # brand-new tree (e.g. a first run against a new *_bigfont output path) rather than one
        # already populated by a prior run. Some individual notebook cells already do this
        # explicitly before calling generate_facebook_post, but not all -- doing it here once
        # covers every caller instead of requiring each cell to remember it.
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"Facebook post generated: {os.path.abspath(output_file)}")

    return html_template


def display_facebook_post(html_content):
    display(HTML(html_content))


