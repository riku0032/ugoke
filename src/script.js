function startCountdown() {
    let count = 5;
    // window.parent (Streamlitの親階層) から要素を取得
    const display = window.parent.document.getElementById('countdown-display');
    
    if (!display) {
        console.log("Waiting for display element...");
        setTimeout(startCountdown, 100);
        return;
    }

    const timer = setInterval(() => {
        if (count > 0) {
            display.innerText = count;
            display.style.transform = "translate(-50%, -50%) scale(1.5)";
            setTimeout(() => { 
                if(display) display.style.transform = "translate(-50%, -50%) scale(1)"; 
            }, 200);
            count--;
        } else {
            display.innerText = "📸";
            clearInterval(timer);
            
            // シャッターボタンを探索してクリック
            const autoClick = () => {
                const buttons = Array.from(window.parent.document.querySelectorAll('button'));
                const shutter = buttons.find(b => 
                    b.innerText.includes('Take Photo') || 
                    b.innerText.includes('写真を撮る') ||
                    b.getAttribute('aria-label')?.includes('Take Photo')
                );
                
                if (shutter) {
                    shutter.click();
                    setTimeout(() => { display.innerText = ""; }, 1000);
                } else {
                    // ボタンが出るまで0.3秒おきにリトライ
                    setTimeout(autoClick, 300);
                }
            };
            autoClick();
        }
    }, 1000);
}
