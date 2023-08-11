import { Directive, ElementRef, Input, OnInit, OnDestroy } from '@angular/core';

@Directive({
  selector: '[appTypingEffect]'
})
export class TypingEffectDirective implements OnInit, OnDestroy {
  @Input('appTypingEffect') typingText: string = '';
  private currentIndex = 0;
  private typingInterval: any;
  private showCursor = true;

  constructor(private el: ElementRef) {}

  ngOnInit(): void {
    this.startTyping();
  }

  private startTyping() {
    this.typingInterval = setInterval(() => {
      if (this.currentIndex <= this.typingText.length) {
        this.el.nativeElement.textContent = this.typingText.slice(0, this.currentIndex);
        this.currentIndex++;
      } else {
        this.showCursor = !this.showCursor;
      }
    }, 100); // Adjust typing speed here (milliseconds)
  }

  ngOnDestroy(): void {
    clearInterval(this.typingInterval);
  }
}
