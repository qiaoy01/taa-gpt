const morseMapping = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----'
};

const audioContext = new (window.AudioContext || window.webkitAudioContext)();

function generateSound(frequency, duration) {
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
    gainNode.gain.setValueAtTime(1, audioContext.currentTime);

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    const endTime = audioContext.currentTime + duration;
    gainNode.gain.setValueAtTime(1, endTime - 0.01);
    gainNode.gain.exponentialRampToValueAtTime(0.001, endTime);

    return { oscillator, endTime };
}


function playMorseCodeString(inputString) {
    const firstChar = inputString.charAt(0);
    const morseCode = morseMapping[firstChar.toUpperCase()];

    const speed = 1.0;

    const dotDuration = 0.1 / speed;
    const dashDuration = 0.3 / speed;
    const pauseDuration = 0.1 / speed;

    let currentTime = audioContext.currentTime;

    for (const c of morseCode) {
        if (c === '.') {
            const { oscillator, endTime } = generateSound(800, dotDuration);
            oscillator.start(currentTime);
            oscillator.stop(endTime);
            currentTime = endTime;
        } else if (c === '-') {
            const { oscillator, endTime } = generateSound(800, dashDuration);
            oscillator.start(currentTime);
            oscillator.stop(endTime);
            currentTime = endTime;
        } else if (c === ' ') {
            currentTime += pauseDuration;
        }
    }
}
