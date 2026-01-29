package com.documiner.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.multipart.MultipartFile;
import java.util.Map;

@FeignClient(name = "ocr-engine", url = "${OCR_ENGINE_URL:http://localhost:5000}")
public interface OcrClient {
    @PostMapping(value = "/process", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    Map<String, String> processFile(@RequestPart("file") MultipartFile file);
}
