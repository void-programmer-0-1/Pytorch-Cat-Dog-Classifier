const CACHE_NAME = "version-1";
const urlsToCache = ["index.html","offline.html"];

const self = this;
let deferredPrompt;

self.addEventListener("install",(event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
        .then((cache) => { console.log("cache added"); return cache.addAll(urlsToCache); })
        .catch((err) => { console.log("Error: ",err)})
    )
});

self.addEventListener("fetch",(event) => {
    event.respondWith(
        caches.match(event.request)
        .then(() => { 
                return fetch(event.request)
                    .catch(() => caches.match("offline.html")) 
        })
    )
});

self.addEventListener("activate",(event) => {
    const cacheWhitelist = [];
    cacheWhitelist.push(CACHE_NAME);
    
    event.waitUntil(
        caches.keys().then((cacheNames) => Promise.all(
                cacheNames.map((cacheName) => {
                    if(!cacheWhitelist.includes(cacheName)){
                        return caches.delete(cacheName);
                    }
                })
            ))
    )
});
