package com.documiner.docmanager.controller;

import com.documiner.docmanager.client.OcrClient;
import com.documiner.docmanager.entity.Invoice;
import com.documiner.docmanager.entity.ProcessingStatus;
import com.documiner.docmanager.repository.InvoiceRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/upload")
@RequiredArgsConstructor
@Slf4j
public class DocumentController {

    private final InvoiceRepository invoiceRepository;
    private final OcrClient ocrClient;

    @PostMapping
    public ResponseEntity<?> uploadDocument(@RequestParam("file") MultipartFile file) {
        log.info("Received file upload request: {}", file.getOriginalFilename());

        Invoice invoice = Invoice.builder()
                .filename(file.getOriginalFilename())
                .status(ProcessingStatus.UPLOADING)
                .build();

        invoice = invoiceRepository.save(invoice);

        try {
            // Call Python Microservice
            log.info("Sending file to OCR Engine...");
            OcrClient.OcrResponse response = ocrClient.processFile(file);
            log.info("OCR Processing successful. Total found: {}", response.getTotal());

            // Update Invoice
            invoice.setExtractedText(response.getText());
            invoice.setTotalAmount(response.getTotal());
            invoice.setConfidenceScore(response.getConfidence());
            invoice.setStatus(ProcessingStatus.COMPLETED);

            invoiceRepository.save(invoice);
            return ResponseEntity.ok(invoice);

        } catch (Exception e) {
            log.error("Error generating OCR for file {}: {}", file.getOriginalFilename(), e.getMessage());
            invoice.setStatus(ProcessingStatus.FAILED);
            invoiceRepository.save(invoice);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error processing document: " + e.getMessage());
        }
    }
}
