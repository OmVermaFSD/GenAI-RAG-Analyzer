package com.documiner.docmanager.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.multipart.MultipartFile;
import lombok.Data;

@FeignClient(name = "ocr-engine", url = "${OCR_ENGINE_URL:http://localhost:5000}")
public interface OcrClient {

    @PostMapping(value = "/process", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    OcrResponse processFile(@RequestPart("file") MultipartFile file);

    @Data
    class OcrResponse {
        private String text;
        private String total;
        private Double confidence;
    }
}
