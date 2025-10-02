
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

