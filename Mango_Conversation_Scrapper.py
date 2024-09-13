import os
import bson
from bson import ObjectId
from collections import defaultdict

# Folder containing your BSON files
folder_path = 'MangoChat/'

# Dictionary to hold all conversations grouped by UserID
conversations_by_user = defaultdict(list)

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.bson'):  # Process only BSON files
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as f:
            # Decode the BSON content
            file_data = bson.decode_all(f.read())
            
            # Group conversations by UserID
            for entry in file_data:
                user_id = entry.get('UserID', 'Unknown User')
                date = entry.get('Date', 'Unknown Date')
                history = entry.get('History', {})
                user_message = history.get('User', 'No message from user')
                response = history.get('ReservoirChat', 'No response from ReservoirChat')
                
                # Store each conversation in the corresponding user_id group
                conversations_by_user[user_id].append({
                    'date': date,
                    'user_message': user_message,
                    'response': response
                })

# Prepare the markdown content
md_content = []

for user_id, conversations in conversations_by_user.items():
    # Create a header for each user
    md_content.append(f"# User: {user_id}\n")
    
    # Add all conversations for this user
    for conversation in conversations:
        date = conversation['date']
        user_message = conversation['user_message']
        response = conversation['response']
        
        # Append each conversation to the markdown content
        md_content.append(f"**Date:** {date}\n")
        md_content.append(f"**User**\n\n{user_message}\n")
        md_content.append(f"**ReservoirChat**\n\n{response}\n")
    
    # Add a separator between users, but not between conversations for the same user
    md_content.append("\n---\n")

# Join all the parts into a single markdown string
markdown_output = "\n".join(md_content)

# Write the markdown content to a .md file
with open('combined_conversations.md', 'w') as md_file:
    md_file.write(markdown_output)

# Optionally, print the content to verify
print(markdown_output)
