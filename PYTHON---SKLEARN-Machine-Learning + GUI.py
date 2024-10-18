import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to run the incremental training
def run_training():
    try:
        # Get the values from the GUI
        num_chunks = int(num_chunks_entry.get())
        max_iter = int(max_iter_entry.get())

        # Generate a random dataset
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize a model that supports incremental learning
        model = SGDClassifier(max_iter=max_iter, tol=1e-3)

        # Split training data into smaller chunks to simulate streaming data
        chunks = np.array_split(X_train, num_chunks)
        chunk_labels = np.array_split(y_train, num_chunks)

        # Train incrementally
        for i in range(len(chunks)):
            X_chunk, y_chunk = chunks[i], chunk_labels[i]
            model.partial_fit(X_chunk, y_chunk, classes=np.unique(y))

            # After every chunk, evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy after chunk {i+1}: {accuracy:.4f}")
            status_label.config(text=f"Accuracy after chunk {i+1}: {accuracy:.4f}")

        # Final evaluation on the test set
        final_accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Final accuracy: {final_accuracy:.4f}")
        status_label.config(text=f"Final accuracy: {final_accuracy:.4f}")
        messagebox.showinfo("Training Complete", f"Final Accuracy: {final_accuracy:.4f}")
        
    except ValueError as e:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for the settings.")

# Set up the GUI using tkinter
root = tk.Tk()
root.title("Incremental Learning Settings")

# Create the labels and entry widgets
tk.Label(root, text="Number of Chunks:").grid(row=0, column=0, padx=10, pady=5)
num_chunks_entry = tk.Entry(root)
num_chunks_entry.grid(row=0, column=1, padx=10, pady=5)
num_chunks_entry.insert(0, "10")  # Default value

tk.Label(root, text="Max Iterations:").grid(row=1, column=0, padx=10, pady=5)
max_iter_entry = tk.Entry(root)
max_iter_entry.grid(row=1, column=1, padx=10, pady=5)
max_iter_entry.insert(0, "1000")  # Default value

# Create a status label for updates
status_label = tk.Label(root, text="Status: Waiting to start training")
status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Create a button to run the training
train_button = tk.Button(root, text="Start Training", command=run_training)
train_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
