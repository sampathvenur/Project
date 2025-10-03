
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

	// Handle API requests by proxying them to the Python backend
	http.HandleFunc("/predict_image", func(w http.ResponseWriter, r *http.Request) {
		proxyToPythonAPI(w, r, "http://localhost:8000/predict_image")
	})

	// Start the server
	http.ListenAndServe(":8080", nil)
}

func proxyToPythonAPI(w http.ResponseWriter, r *http.Request, apiURL string) {
	// Handle only POST requests
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse the multipart form
	if err := r.ParseMultipartForm(10 << 20); err != nil { // 10 MB
		http.Error(w, "Error parsing multipart form", http.StatusBadRequest)
		return
	}

	// Get the file from the form
	file, handler, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Error retrieving the file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Get the expected_type from the form
	expectedType := r.FormValue("expected_type")

	// Create a new multipart writer
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Create and copy the file part
	part, err := writer.CreateFormFile("file", handler.Filename)
	if err != nil {
		http.Error(w, "Error creating form file", http.StatusInternalServerError)
		return
	}
	if _, err = io.Copy(part, file); err != nil {
		http.Error(w, "Error copying file content", http.StatusInternalServerError)
		return
	}

	// Create the expected_type form field
	if err = writer.WriteField("expected_type", expectedType); err != nil {
		http.Error(w, "Error writing form field", http.StatusInternalServerError)
		return
	}

	writer.Close()

	// Create a new request to the Python API
	req, err := http.NewRequest("POST", apiURL, body)
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
