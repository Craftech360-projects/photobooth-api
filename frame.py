# from PIL import Image

# # Load the frame and the final image
# frame = Image.open('uploads/frame.jpg')
# final = Image.open('results/final.jpg')

# # Ensure final image is the correct size (900x1200)
# if final.size != (900, 1200):
#     final = final.resize((900, 1200), Image.ANTIALIAS)

# # Calculate the position to paste the final image
# # We want the top of final.jpg to be 240 pixels from the top of frame.jpg
# x = (frame.width - final.width) // 2  # Center horizontally
# y = 280  # 240 pixels from the top

# # Paste final.jpg onto frame.jpg at the calculated position
# frame.paste(final, (x, y))

# # Save the result
# frame.save('uploads/framed_final.jpg')

# print("Image processing complete. Check 'uploads/framed_final.jpg' for the result.")


from PIL import Image

# Load the frame and the final image
frame = Image.open('uploads/frame.jpg')
final = Image.open('results/final.jpg')

# Scale factor to make the final image bigger (e.g., 1.1 for 10% larger, 1.2 for 20% larger)
scale_factor = .83# Adjust this value as needed
new_width = int(final.width * scale_factor)
new_height = int(final.height * scale_factor)

# Resize the final image to the new dimensions
final = final.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Calculate the position to paste the final image
# We want the top of final.jpg to be 240 pixels from the top of frame.jpg
x = (frame.width - final.width) // 2  # Center horizontally
y =  210   # 240 pixels from the top
  
# Paste final.jpg onto frame.jpg at the calculated position
frame.paste(final, (x, y))

# Save the result
frame.save('uploads/framed_final.jpg')

print("Image processing complete. Check 'uploads/framed_final.jpg' for the result.")