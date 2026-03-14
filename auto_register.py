import subprocess

print("=== STUDENT REGISTRATION ===")

# Step 1: Capture photos
print("Opening camera... Take 25-30 photos")
subprocess.run(["python", "capture_faces.py"])

# Step 2: Generate extra images
print("Creating AI images...")
subprocess.run(["python", "augment_faces.py"])

# Step 3: Train system
print("Training face model...")
subprocess.run(["python", "generate_embeddings.py"])

print("DONE ✅ Student registered successfully!")
