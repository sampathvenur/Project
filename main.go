
package main

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
)

func main() {
	// Serve the frontend
	fs := http.FileServer(http.Dir("./web"))
	http.Handle("/", fs)

	// Handle the prediction request
	http.HandleFunc("/predict", handlePredict)

	// Start the server
	http.ListenAndServe(":8080", nil)
}

func handlePredict(w http.ResponseWriter, r *http.Request) {
	// Parse the multipart form
	r.ParseMultipartForm(10 << 20) // 10 MB
	file, handler, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Error retrieving the file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Create a new multipart writer
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", handler.Filename)
	if err != nil {
		http.Error(w, "Error creating form file", http.StatusInternalServerError)
		return
	}

	// Copy the file content
	_, err = io.Copy(part, file)
	if err != nil {
		http.Error(w, "Error copying file content", http.StatusInternalServerError)
		return
	}
	writer.Close()

	// Create a new request to the Python API
	req, err := http.NewRequest("POST", "http://localhost:8000/predict", body)
	if err != nil {
		http.Error(w, "Error creating request to Python API", http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "Error sending request to Python API", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// Read the response
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "Error reading response from Python API", http.StatusInternalServerError)
		return
	}

	// Send the response back to the frontend
	w.Header().Set("Content-Type", "application/json")
	w.Write(responseBody)
}
