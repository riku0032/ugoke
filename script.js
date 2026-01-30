function startCountdown() {
    let count = 5;
    const display = document.getElementById('countdown-display');
    if (!display) return;
    
    const timer = setInterval(() => {
        if (count > 0) {
            display.innerText = count;
            display.style.transform = "scale(1.5)";
            setTimeout(() => { if(display) display.style.transform = "scale(1)"; }, 200);
            count--;
        } else {
            display.innerText = "📸";
            clearInterval(timer);
            
            // Streamlitのシャッターボタンを自動クリック
            // ブラウザの言語設定により「Take Photo」か「写真を撮る」を探す
            const buttons = window.parent.document.querySelectorAll('button');
            const photoButton = Array.from(buttons).find(b => 
                b.innerText.includes('Take Photo') || 
                b.innerText.includes('写真を撮る') ||
                b.innerText.includes('shutter')
            );
            if (photoButton) {
                photoButton.click();
            }
        }
    }, 1000);
}