const CACHE_NAME = 'demucs-model-v1';
const MODEL_URL_PATTERN = /huggingface\.co.*\.onnx/;

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) => {
      return Promise.all(
        names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name))
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = event.request.url;
  
  // Only cache ONNX model files from HuggingFace
  if (MODEL_URL_PATTERN.test(url)) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) {
          console.log('[SW] Model served from cache:', url.split('/').pop());
          return cached;
        }
        
        return fetch(event.request).then((response) => {
          // Only cache successful responses
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, clone);
              console.log('[SW] Model cached:', url.split('/').pop());
            });
          }
          return response;
        });
      })
    );
    return;
  }
  
  // Pass through all other requests
  return;
});
