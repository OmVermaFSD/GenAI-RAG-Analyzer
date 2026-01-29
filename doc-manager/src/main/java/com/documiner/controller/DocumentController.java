package com.documiner.controller;

import com.documiner.client.OcrClient;
import com.documiner.entity.Invoice;
import com.documiner.repository.InvoiceRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.Map;

@RestController
@RequiredArgsConstructor
public class DocumentController {

    private final InvoiceRepository invoiceRepository;
    private final OcrClient ocrClient;

    @PostMapping("/upload")
    public Invoice uploadDocument(@RequestParam("file") MultipartFile file) {
        Invoice invoice = new Invoice();
        invoice.setFilename(file.getOriginalFilename());

        try {
            Map<String, String> result = ocrClient.processFile(file);
            invoice.setExtractedText(result.get("text"));
        } catch (Exception e) {
            invoice.setExtractedText("Error: " + e.getMessage());
        }

        return invoiceRepository.save(invoice);
    }
}
